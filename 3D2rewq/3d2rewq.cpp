#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "sys/time.h"

#include <mpi.h>
#include <omp.h>

#include "global.h"
#include "helpers.h"



#define PIE 3.1415926   // [Afa] Delicious fruit pie

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int initFlag;
    MPI_Initialized(&initFlag);
    if (!initFlag) {
        printf("MPI init failed\n");
        return EXIT_FAILURE;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int i,j,k,kk,kkk,l,mm=5;
    int nx,ny,nz,lt,nedge;
    int nleft,nright,nfront,nback,ntop,nbottom;
    float frequency;
    float velmax;
    float dt;
    int ncx_shot1,ncy_shot1,ncz_shot;
    int ishot,ncy_shot,ncx_shot;
    float unit;
    int nxshot,nyshot,dxshot,dyshot;
    char infile[80],outfile[80],logfile[80],tmp[80];
    FILE  *fin, *fout, *flog;
    MPI_File mpi_flog, mpi_fout;
    MPI_Status mpi_status;
    struct timeval start,end;
    float all_time;

    float *u, *v, *w, *up, *up1, *up2,
            *vp, *vp1, *vp2, *wp, *wp1, *wp2,
            *us, *us1, *us2, *vs, *vs1, *vs2,
            *ws, *ws1, *ws2, *vpp, *density, *vss;
    float c[5][7];
    float *wave;
    float nshot,t0,tt,c0;
    float dtx,dtz,dtxz,dr1,dr2,dtx4,dtz4,dtxz4;
    float xmax,px,sx;
    float vvp2,drd1,drd2,vvs2, tempux2, tempuy2, tempuz2, tempvx2,
            tempvy2, tempvz2, tempwx2, tempwy2, tempwz2, tempuxz,
            tempuxy, tempvyz, tempvxy, tempwxz, tempwyz;
    char message[100];

    if(argc<4)
    {
        printf("please add 3 parameter: inpurfile, outfile, logfile\n");
        exit(0);
    }

    message[99] = 0;    // Avoid string buffer overrun

    strcpy(infile,argv[1]);
    strcpy(outfile,argv[2]);
    strcpy(logfile,argv[3]);

    strcpy(tmp,"date ");
    strncat(tmp, ">> ",3);
    strncat(tmp, logfile, strlen(logfile));
    if (proc_rank == 0) {
        flog = fopen(logfile,"w");
        fprintf(flog,"------------start time------------\n");
        fclose(flog);
        system(tmp);
        gettimeofday(&start,NULL);
    }
    fin = fopen(infile,"r");
    if(fin == NULL)
    {
        printf("file %s is  not exist\n",infile);
        exit(0);
    }
    fscanf(fin,"nx=%d\n",&nx);
    fscanf(fin,"ny=%d\n",&ny);
    fscanf(fin,"nz=%d\n",&nz);
    fscanf(fin,"lt=%d\n",&lt);
    fscanf(fin,"nedge=%d\n",&nedge);
    fscanf(fin,"ncx_shot1=%d\n",&ncx_shot1);
    fscanf(fin,"ncy_shot1=%d\n",&ncy_shot1);
    fscanf(fin,"ncz_shot=%d\n",&ncz_shot);
    fscanf(fin,"nxshot=%d\n",&nxshot);
    fscanf(fin,"nyshot=%d\n",&nyshot);
    fscanf(fin,"frequency=%f\n",&frequency);
    fscanf(fin,"velmax=%f\n",&velmax);
    fscanf(fin,"dt=%f\n",&dt);
    fscanf(fin,"unit=%f\n",&unit);
    fscanf(fin,"dxshot=%d\n",&dxshot);
    fscanf(fin,"dyshot=%d\n",&dyshot);
    fclose(fin);

    printf("\n--------workload parameter--------\n");
    printf("nx=%d\n",nx);
    printf("ny=%d\n",ny);
    printf("nz=%d\n",nz);
    printf("lt=%d\n",lt);
    printf("nedge=%d\n",nedge);
    printf("ncx_shot1=%d\n",ncx_shot1);
    printf("ncy_shot1=%d\n",ncy_shot1);
    printf("ncz_shot=%d\n",ncz_shot);
    printf("nxshot=%d\n",nxshot);
    printf("nyshot=%d\n",nyshot);
    printf("frequency=%f\n",frequency);
    printf("velmax=%f\n",velmax);
    printf("dt=%f\n",dt);
    printf("unit=%f\n",unit);
    printf("dxshot=%d\n",dxshot);
    printf("dyshot=%d\n\n",dyshot);
    if (proc_rank == 0) {   // Master
        flog = fopen(logfile,"a");
        fprintf(flog,"\n--------workload parameter--------\n");
        fprintf(flog,"nx=%d\n",nx);
        fprintf(flog,"ny=%d\n",ny);
        fprintf(flog,"nz=%d\n",nz);
        fprintf(flog,"lt=%d\n",lt);
        fprintf(flog,"nedge=%d\n",nedge);
        fprintf(flog,"ncx_shot1=%d\n",ncx_shot1);
        fprintf(flog,"ncy_shot1=%d\n",ncy_shot1);
        fprintf(flog,"ncz_shot=%d\n",ncz_shot);
        fprintf(flog,"nxshot=%d\n",nxshot);
        fprintf(flog,"nyshot=%d\n",nyshot);
        fprintf(flog,"frequency=%f\n",frequency);
        fprintf(flog,"velmax=%f\n",velmax);
        fprintf(flog,"dt=%f\n",dt);
        fprintf(flog,"unit=%f\n",unit);
        fprintf(flog,"dxshot=%d\n",dxshot);
        fprintf(flog,"dyshot=%d\n\n",dyshot);
        fclose(flog);
    }

#ifdef _WITH_PHI
    // [Afa] It is recommended that for Intel Xeon Phi data is 64-byte aligned.
    // Upon successful completion, posix_memalign() shall return zero
    if (posix_memalign((void **)&u  , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&v  , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&w  , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&up , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&up1, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&up2, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&vp , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&vp1, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&vp2, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&wp , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&wp1, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&wp2, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&us , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&us1, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&us2, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&vs , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&vs1, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&vs2, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&ws , 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&ws1, 64, sizeof(float)*nz*ny*nx)) return 2;
    if (posix_memalign((void **)&ws2, 64, sizeof(float)*nz*ny*nx)) return 2;
#else
    u       = (float*)malloc(sizeof(float)*nz*ny*nx);
    v       = (float*)malloc(sizeof(float)*nz*ny*nx);
    w       = (float*)malloc(sizeof(float)*nz*ny*nx);
    up      = (float*)malloc(sizeof(float)*nz*ny*nx);
    up1     = (float*)malloc(sizeof(float)*nz*ny*nx);
    up2     = (float*)malloc(sizeof(float)*nz*ny*nx);
    vp      = (float*)malloc(sizeof(float)*nz*ny*nx);
    vp1     = (float*)malloc(sizeof(float)*nz*ny*nx);
    vp2     = (float*)malloc(sizeof(float)*nz*ny*nx);
    wp      = (float*)malloc(sizeof(float)*nz*ny*nx);
    wp1     = (float*)malloc(sizeof(float)*nz*ny*nx);
    wp2     = (float*)malloc(sizeof(float)*nz*ny*nx);
    us      = (float*)malloc(sizeof(float)*nz*ny*nx);
    us1     = (float*)malloc(sizeof(float)*nz*ny*nx);
    us2     = (float*)malloc(sizeof(float)*nz*ny*nx);
    vs      = (float*)malloc(sizeof(float)*nz*ny*nx);
    vs1     = (float*)malloc(sizeof(float)*nz*ny*nx);
    vs2     = (float*)malloc(sizeof(float)*nz*ny*nx);
    ws      = (float*)malloc(sizeof(float)*nz*ny*nx);
    ws1     = (float*)malloc(sizeof(float)*nz*ny*nx);
    ws2     = (float*)malloc(sizeof(float)*nz*ny*nx);
#endif
    // [Afa] Those are not offloaded to phi yet
    vpp     = (float*)malloc(sizeof(float)*nz*ny*nx);
    density = (float*)malloc(sizeof(float)*nz*ny*nx);
    vss     = (float*)malloc(sizeof(float)*nz*ny*nx);
    wave = (float*)malloc(sizeof(float)*lt);

    nshot=nxshot*nyshot;
    t0=1.0/frequency;

    // [Afa] Branch optmization
    // TODO: Will compiler optimize the `condition'?
    //       i.e Can I write `for(i=0;i< (nz < 210 ? nz : 210);i++)'?
    int condition = nz < 210 ? nz : 210;
    for(i=0; i < condition;i++) {
        for(j=0;j<ny;j++) {
            for(k=0;k<nx;k++) {
                vpp[i*ny*nx+j*nx+k]=2300.;
                vss[i*ny*nx+j*nx+k]=1232.;
                density[i*ny*nx+j*nx+k]=1.;
            }
        }
    }

    condition = i < (nz < 260 ? nz : 260);
    for(i=210; i < condition;i++) {
        for(j=0;j<ny;j++) {
            for(k=0;k<nx;k++) {
                vpp[i*ny*nx+j*nx+k]=2800.;
                vss[i*ny*nx+j*nx+k]=1509.;
                density[i*ny*nx+j*nx+k]=2.;
            }
        }
    }

    for(i=260;i<nz;i++) {
        for(j=0;j<ny;j++) {
            for(k=0;k<nx;k++)
            {
                vpp[i*ny*nx+j*nx+k]=3500.;
                vss[i*ny*nx+j*nx+k]=1909.;
                density[i*ny*nx+j*nx+k]=2.5;
            }
        }
    }

    for(l=0;l<lt;l++)
    {
        tt=l*dt;
        tt=tt-t0;
        float sp=PIE*frequency*tt;
        float fx=100000.*exp(-sp*sp)*(1.-2.*sp*sp);
        wave[l]=fx;
    }

    // TODO: [Afa] Data produced by code below are static. See table below
    if(mm==5)
    {
        c0=-2.927222164;
        c[0][0]=1.66666665;
        c[1][0]=-0.23809525;
        c[2][0]=0.03968254;
        c[3][0]=-0.004960318;
        c[4][0]=0.0003174603;
    }

    c[0][1]=0.83333;
    c[1][1]=-0.2381;
    c[2][1]=0.0595;
    c[3][1]=-0.0099;
    c[4][1]=0.0008;

    for(i=0;i<5;i++)
        for(j=0;j<5;j++)
            c[j][2+i]=c[i][1]*c[j][1];
    /*
     * mm == 5, c =
     * 1.666667    0.833330    0.694439    -0.198416   0.049583    -0.008250   0.000667
     * -0.238095   -0.238100   -0.198416   0.056692    -0.014167   0.002357    -0.000190
     * 0.039683    0.059500    0.049583    -0.014167   0.003540    -0.000589   0.000048
     * -0.004960   -0.009900   -0.008250   0.002357    -0.000589   0.000098    -0.000008
     * 0.000317    0.000800    0.000667    -0.000190   0.000048    -0.000008   0.000001
    */

    /*
     * mm != 5, c =
     * 0.000000    0.833330    0.694439    -0.198416   0.049583    -0.008250   0.000667
     * 0.000000    -0.238100   -0.198416   0.056692    -0.014167   0.002357    -0.000190
     * 0.000000    0.059500    0.049583    -0.014167   0.003540    -0.000589   0.000048
     * 0.000000    -0.009900   -0.008250   0.002357    -0.000589   0.000098    -0.000008
     * 0.000000    0.000800    0.000667    -0.000190   0.000048    -0.000008   0.000001
     */

    dtx=dt/unit;
    dtz=dt/unit;
    dtxz=dtx*dtz;

    dr1=dtx*dtx/2.;
    dr2=dtz*dtz/2.;

    dtx4=dtx*dtx*dtx*dtx;
    dtz4=dtz*dtz*dtz*dtz;
    dtxz4=dtx*dtx*dtz*dtz;

    if (proc_rank == 0) {
        fout = fopen(outfile, "wb");
        fclose(fout);
    }   // [Afa] Truncate file. We need a prettier way

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_APPEND | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_fout);
    MPI_File_open(MPI_COMM_WORLD, logfile, MPI_MODE_APPEND | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_flog);

    // [Afa] *About Nodes Number* nshot (i.e nxshot * nyshot) should be multiple of node numbers,
    //       or there will be hungry processes
    int loop_per_proc = ((int)nshot % world_size == 0) ? (nshot / world_size) : (nshot / world_size + 1);
    //    for(ishot=1;ishot<=nshot;ishot++)   // [Afa] nshot is 20 in para1.in, but 200 in para2.in
    for (int loop_index = 0; loop_index < loop_per_proc; ++loop_index)
    {
        ishot = loop_index + proc_rank * loop_per_proc + 1; // [Afa] See commented code 2 lines above to understand this line
        if (ishot <= nshot) { // [Afa] ishot <= nshot
            printf("shot=%d, process %d\n",ishot, proc_rank);
            snprintf(message, 99, "shot=%d, process %d\n", ishot, proc_rank);
            MPI_File_write(mpi_flog, message, strlen(message), MPI_CHAR, &mpi_status);
        } else {
            printf("shot=HUNGRY, process %d\n",proc_rank);
            snprintf(message, 99, "shot=HUNGRY, process %d\n", proc_rank);
            MPI_File_write(mpi_flog, message, strlen(message), MPI_CHAR, &mpi_status);
            continue;
        }
        ncy_shot=ncy_shot1+(ishot/nxshot)*dyshot;
        ncx_shot=ncx_shot1+(ishot%nxshot)*dxshot;

        // [Afa] Matrix is zeroed in every loop
        // i.e. The relation between those matrices in each loop is pretty loose
        // Matrices not zeroed are: vpp, density, vss and wave, and they're not changed (read-only)
        // We only need to partially collect matrix `up'

        // TODO: [Afa] Get a better way to pass those pointers, and mark them as `restrict'
        // And WHY are they using cpp as extension? C++11 doesn't support `restrict'
        zero_matrices(u, w, ws2, up2, vp1, wp1, us, ws, wp, us2, us1, wp2,
                      v, up1, nz, nx, up, ny, ws1, vs, vp2, vs1, vs2, vp);

        for(l=1;l<=lt;l++)
        {
            xmax=l*dt*velmax;
            nleft=ncx_shot-xmax/unit-10;
            nright=ncx_shot+xmax/unit+10;
            nfront=ncy_shot-xmax/unit-10;
            nback=ncy_shot+xmax/unit+10;
            ntop=ncz_shot-xmax/unit-10;
            nbottom=ncz_shot+xmax/unit+10;
            if(nleft<5) nleft=5;
            if(nright>nx-5) nright=nx-5;
            if(nfront<5) nfront=5;
            if(nback>ny-5) nback=ny-5;
            if(ntop<5) ntop=5;
            if(nbottom>nz-5) nbottom=nz-5;
            ntop = ntop-1;
            nfront = nfront-1;
            nleft = nleft-1;

            for(k=ntop;k<nbottom;k++)
                for(j=nfront;j<nback;j++)
                    for(i=nleft;i<nright;i++)
                    {
                        if(i==ncx_shot-1&&j==ncy_shot-1&&k==ncz_shot-1)
                        {
                            px=1.;
                            sx=0.;
                        }
                        else
                        {
                            px=0.;
                            sx=0.;
                        }
                        vvp2=vpp[k*ny*nx+j*nx+i]*vpp[k*ny*nx+j*nx+i];
                        drd1=dr1*vvp2;
                        drd2=dr2*vvp2;

                        vvs2=vss[k*ny*nx+j*nx+i]*vss[k*ny*nx+j*nx+i];
                        drd1=dr1*vvs2;
                        drd2=dr2*vvs2;

                        tempux2=0.0f;
                        tempuy2=0.0f;
                        tempuz2=0.0f;
                        tempvx2=0.0f;
                        tempvy2=0.0f;
                        tempvz2=0.0f;
                        tempwx2=0.0f;
                        tempwy2=0.0f;
                        tempwz2=0.0f;
                        tempuxz=0.0f;
                        tempuxy=0.0f;
                        tempvyz=0.0f;
                        tempvxy=0.0f;
                        tempwxz=0.0f;
                        tempwyz=0.0f;
                        for(kk=1;kk<=mm;kk++)
                        {
                            tempux2=tempux2+c[kk-1][0]*(u[k*ny*nx+j*nx+(i+kk)]+u[k*ny*nx+j*nx+(i-kk)]);
                            tempuy2=tempuy2+c[kk-1][0]*(u[k*ny*nx+(j+kk)*nx+i]+u[k*ny*nx+(j-kk)*nx+i]);
                            tempuz2=tempuz2+c[kk-1][0]*(u[(k+kk)*ny*nx+j*nx+i]+u[(k-kk)*ny*nx+j*nx+i]);

                            tempvx2=tempvx2+c[kk-1][0]*(v[k*ny*nx+j*nx+(i+kk)]+v[k*ny*nx+j*nx+(i-kk)]);
                            tempvy2=tempvy2+c[kk-1][0]*(v[k*ny*nx+(j+kk)*nx+i]+v[k*ny*nx+(j-kk)*nx+i]);
                            tempvz2=tempvz2+c[kk-1][0]*(v[(k+kk)*ny*nx+j*nx+i]+v[(k-kk)*ny*nx+j*nx+i]);

                            tempwx2=tempwx2+c[kk-1][0]*(w[k*ny*nx+j*nx+(i+kk)]+w[k*ny*nx+j*nx+(i-kk)]);
                            tempwy2=tempwy2+c[kk-1][0]*(w[k*ny*nx+(j+kk)*nx+i]+w[k*ny*nx+(j-kk)*nx+i]);
                            tempwz2=tempwz2+c[kk-1][0]*(w[(k+kk)*ny*nx+j*nx+i]+w[(k-kk)*ny*nx+j*nx+i]);

                        } //for(kk=1;kk<=mm;kk++) end

                        tempux2=(tempux2+c0*u[k*ny*nx+j*nx+i])*vvp2*dtx*dtx;
                        tempuy2=(tempuy2+c0*u[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempuz2=(tempuz2+c0*u[k*ny*nx+j*nx+i])*vvs2*dtz*dtz;

                        tempvx2=(tempvx2+c0*v[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempvy2=(tempvy2+c0*v[k*ny*nx+j*nx+i])*vvp2*dtx*dtx;
                        tempvz2=(tempvz2+c0*v[k*ny*nx+j*nx+i])*vvs2*dtz*dtz;

                        tempwx2=(tempwx2+c0*w[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempwy2=(tempwy2+c0*w[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempwz2=(tempwz2+c0*w[k*ny*nx+j*nx+i])*vvp2*dtz*dtz;

                        for(kk=1;kk<=mm;kk++)
                        {
                            for(kkk=1;kkk<=mm;kkk++)
                            {
                                tempuxz=tempuxz+c[kkk-1][1+kk]*(u[(k+kkk)*ny*nx+j*nx+(i+kk)]
                                        -u[(k-kkk)*ny*nx+j*nx+(i+kk)]
                                        +u[(k-kkk)*ny*nx+j*nx+(i-kk)]
                                        -u[(k+kkk)*ny*nx+j*nx+(i-kk)]);
                                tempuxy=tempuxy+c[kkk-1][1+kk]*(u[k*ny*nx+(j+kkk)*nx+(i+kk)]
                                        -u[k*ny*nx+(j-kkk)*nx+(i+kk)]
                                        +u[k*ny*nx+(j-kkk)*nx+(i-kk)]
                                        -u[k*ny*nx+(j+kkk)*nx+(i-kk)]);

                                tempvyz=tempvyz+c[kkk-1][1+kk]*(v[(k+kkk)*ny*nx+(j+kk)*nx+i]
                                        -v[(k-kkk)*ny*nx+(j+kk)*nx+i]
                                        +v[(k-kkk)*ny*nx+(j-kk)*nx+i]
                                        -v[(k+kkk)*ny*nx+(j-kk)*nx+i]);
                                tempvxy=tempvxy+c[kkk-1][1+kk]*(v[k*ny*nx+(j+kkk)*nx+(i+kk)]
                                        -v[k*ny*nx+(j-kkk)*nx+(i+kk)]
                                        +v[k*ny*nx+(j-kkk)*nx+(i-kk)]
                                        -v[k*ny*nx+(j+kkk)*nx+(i-kk)]);

                                tempwyz=tempwyz+c[kkk-1][1+kk]*(w[(k+kkk)*ny*nx+(j+kk)*nx+i]
                                        -w[(k-kkk)*ny*nx+(j+kk)*nx+i]
                                        +w[(k-kkk)*ny*nx+(j-kk)*nx+i]
                                        -w[(k+kkk)*ny*nx+(j-kk)*nx+i]);
                                tempwxz=tempwxz+c[kkk-1][1+kk]*(w[(k+kkk)*ny*nx+j*nx+(i+kk)]
                                        -w[(k-kkk)*ny*nx+j*nx+(i+kk)]
                                        +w[(k-kkk)*ny*nx+j*nx+(i-kk)]
                                        -w[(k+kkk)*ny*nx+j*nx+(i-kk)]);
                            } // for(kkk=1;kkk<=mm;kkk++) end
                        } //for(kk=1;kk<=mm;kk++) end
                        up[k*ny*nx+j*nx+i]=2.*up1[k*ny*nx+j*nx+i]-up2[k*ny*nx+j*nx+i]
                                +tempux2+tempwxz*vvp2*dtz*dtx
                                +tempvxy*vvp2*dtz*dtx;
                        vp[k*ny*nx+j*nx+i]=2.*vp1[k*ny*nx+j*nx+i]-vp2[k*ny*nx+j*nx+i]
                                +tempvy2+tempuxy*vvp2*dtz*dtx
                                +tempwyz*vvp2*dtz*dtx;
                        wp[k*ny*nx+j*nx+i]=2.*wp1[k*ny*nx+j*nx+i]-wp2[k*ny*nx+j*nx+i]
                                +tempwz2+tempuxz*vvp2*dtz*dtx
                                +tempvyz*vvp2*dtz*dtx
                                +px*wave[l-1];
                        us[k*ny*nx+j*nx+i]=2.*us1[k*ny*nx+j*nx+i]-us2[k*ny*nx+j*nx+i]+tempuy2+tempuz2
                                -tempvxy*vvs2*dtz*dtx-tempwxz*vvs2*dtz*dtx;
                        vs[k*ny*nx+j*nx+i]=2.*vs1[k*ny*nx+j*nx+i]-vs2[k*ny*nx+j*nx+i]+tempvx2+tempvz2
                                -tempuxy*vvs2*dtz*dtx-tempwyz*vvs2*dtz*dtx;
                        ws[k*ny*nx+j*nx+i]=2.*ws1[k*ny*nx+j*nx+i]-ws2[k*ny*nx+j*nx+i]+tempwx2+tempwy2
                                -tempuxz*vvs2*dtz*dtx-tempvyz*vvs2*dtz*dtx;
                    }//for(i=nleft;i<nright;i++) end
            for(k=ntop;k<nbottom;k++)
                for(j=nfront;j<nback;j++)
                    for(i=nleft;i<nright;i++)
                    {
                        u[k*ny*nx+j*nx+i]=up[k*ny*nx+j*nx+i]+us[k*ny*nx+j*nx+i];
                        v[k*ny*nx+j*nx+i]=vp[k*ny*nx+j*nx+i]+vs[k*ny*nx+j*nx+i];
                        w[k*ny*nx+j*nx+i]=wp[k*ny*nx+j*nx+i]+ws[k*ny*nx+j*nx+i];

                        up2[k*ny*nx+j*nx+i]=up1[k*ny*nx+j*nx+i];
                        up1[k*ny*nx+j*nx+i]=up[k*ny*nx+j*nx+i];
                        us2[k*ny*nx+j*nx+i]=us1[k*ny*nx+j*nx+i];
                        us1[k*ny*nx+j*nx+i]=us[k*ny*nx+j*nx+i];
                        vp2[k*ny*nx+j*nx+i]=vp1[k*ny*nx+j*nx+i];
                        vp1[k*ny*nx+j*nx+i]=vp[k*ny*nx+j*nx+i];
                        vs2[k*ny*nx+j*nx+i]=vs1[k*ny*nx+j*nx+i];
                        vs1[k*ny*nx+j*nx+i]=vs[k*ny*nx+j*nx+i];
                        wp2[k*ny*nx+j*nx+i]=wp1[k*ny*nx+j*nx+i];
                        wp1[k*ny*nx+j*nx+i]=wp[k*ny*nx+j*nx+i];
                        ws2[k*ny*nx+j*nx+i]=ws1[k*ny*nx+j*nx+i];
                        ws1[k*ny*nx+j*nx+i]=ws[k*ny*nx+j*nx+i];
                    }//for(i=nleft;i<nright;i++) end
        }//for(l=1;l<=lt;l++) end
        // [Afa] Do we need to keep the order of data?
        //        fwrite(up+169*ny*nx,sizeof(float),ny*nx,fout);    // This is the original fwrite

        MPI_File_write(mpi_fout, up+169*ny*nx, ny * nx, MPI_FLOAT, &mpi_status);

    }//for(ishot=1;ishot<=nshot;ishot++) end
    if (proc_rank == 0) {
        fclose(fout);
    }

    free(u);
    free(v);
    free(w);
    free(up);
    free(up1);
    free(up2);
    free(vp);
    free(vp1);
    free(vp2);
    free(wp);
    free(wp1);
    free(wp2);
    free(us);
    free(us1);
    free(us2);
    free(vs);
    free(vs1);
    free(vs2);
    free(ws);
    free(ws1);
    free(ws2);
    free(vpp);
    free(density);
    free(vss);
    free(wave);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    if (proc_rank == 0) {
        gettimeofday(&end,NULL);
        all_time = (end.tv_sec-start.tv_sec)+(float)(end.tv_usec-start.tv_usec)/1000000.0;
        printf("run time:\t%f s\n",all_time);
        flog = fopen(logfile,"a");
        fprintf(flog,"\nrun time:\t%f s\n\n",all_time);
        fclose(flog);
        flog = fopen(logfile,"a");
        fprintf(flog,"------------end time------------\n");
        fclose(flog);
        system(tmp);
    }


    // Why return 1?
    return 1;
}










