#include <mpi.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "sys/time.h"
#include <unistd.h>

#include <omp.h>
#include <mkl.h>
#include <immintrin.h>

#define PIE 3.1415926   // [Afa] Delicious fruit pie
#define MPI_LOG_TAG 10
#define MPI_RESULT_TAG 20

int proc_rank;
int world_size;

void zero_matrices(float *u, float *w, float *ws2, float *up2, float *vp1, float *wp1, float *us, float *ws, float *wp,
                   float *us2, float *us1, float *wp2, float *v, float *up1, int nz, int nx, float *up,
                   int ny, float *ws1, float *vs, float *vp2, float *vs1, float *vs2, float *vp);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int initFlag;
    MPI_Initialized(&initFlag);
    if (!initFlag) {
        printf("MPI init failed\n");
        return 8;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int l,mm=5;
    int nx,ny,nz,lt,nedge;
    float frequency;
    float velmax;
    float dt;
    int ncx_shot1,ncy_shot1,ncz_shot;
    int ishot,ncy_shot,ncx_shot;
    float unit;
    int nxshot,nyshot,dxshot,dyshot;
    char infile[80],outfile[80],logfile[80],tmp[80], nodelog[84];
    FILE  *fin, *fout, *flog, *fnode;
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
    char message[100];

    if(argc<4)
    {
        printf("please add 3 parameter: inpurfile, outfile, logfile\n");
        exit(1);
    }

    message[99] = 0;    // Avoid string buffer overrun

    strcpy(infile,argv[1]);
    strcpy(outfile,argv[2]);
    strcpy(logfile,argv[3]);
    strcpy(nodelog,logfile);
    strcat(nodelog, ".node");

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
        exit(2);
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
    if (proc_rank == 0) {   // Master
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
        fnode = fopen(nodelog, "a");
        fprintf(fnode,"World size: %d\n", world_size);
        fclose(fnode);
    }

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
    for(int i=0; i < condition;i++) {
        for(int j=0;j<ny;j++) {
            for(int k=0;k<nx;k++) {
                vpp[i*ny*nx+j*nx+k]=2300.;
                vss[i*ny*nx+j*nx+k]=1232.;
                density[i*ny*nx+j*nx+k]=1.;
            }
        }
    }

    condition = nz < 260 ? nz : 260;
    for(int i=210; i < condition;i++) {
        for(int j=0;j<ny;j++) {
            for(int k=0;k<nx;k++) {
                vpp[i*ny*nx+j*nx+k]=2800.;
                vss[i*ny*nx+j*nx+k]=1509.;
                density[i*ny*nx+j*nx+k]=2.;
            }
        }
    }

    for(int i=260;i<nz;i++) {
        for(int j=0;j<ny;j++) {
            for(int k=0;k<nx;k++)
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

    for(int i=0;i<5;i++)
        for(int j=0;j<5;j++)
            c[j][2+i]=c[i][1]*c[j][1];

    float c_col0[]      __attribute__((aligned(16)))
            = {1.66666665, -0.23809525, 0.03968254, -0.004960318, 0.0003174603};
    float c_col0_sum[]  __attribute__((aligned(16)))
            = { 1.66666665, -0.23809525, 0.03968254, -0.004960318, 0.0003174603,
                -2.927222164,
                0.0003174603, -0.004960318, 0.03968254, -0.23809525, 1.66666665
              };
    if (mm!=5) {memset(c_col0, 0, 5 * sizeof(float)); memset(c_col0_sum, 0, 11 * sizeof(float));}
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
    MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_fout);
    MPI_File_open(MPI_COMM_WORLD, nodelog, MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_flog);
    // [Afa] *About Nodes Number* nshot (i.e nxshot * nyshot) should be multiple of node numbers,
    //       or there will be hungry processes
    int loop_per_proc = ((int)nshot % world_size == 0) ? (nshot / world_size) : (nshot / world_size + 1);
    printf("\x1B[31mDEBUG:\x1b[39;49m World size %d, Loop per Proc %d, nshot %f, I am No. %d\n",
           world_size, loop_per_proc, nshot, proc_rank);

    //    for(ishot=1;ishot<=nshot;ishot++)   // [Afa] nshot is 20 in para1.in, but 200 in para2.in
    for (int loop_index = 0; loop_index < loop_per_proc; ++loop_index)
    {
        ishot = loop_index + proc_rank * loop_per_proc + 1; // [Afa] See commented code 2 lines above to understand this line
        if (ishot <= nshot) { // [Afa] ishot <= nshot
            printf("shot %d, process %d\n",ishot, proc_rank);
            snprintf(message, 29, "shot %6d, process %6d\n", ishot, proc_rank);     // [Afa] Those numbers:
            MPI_File_seek(mpi_flog, 28 * (ishot - 1), MPI_SEEK_SET);                // 28: string without '\0'
            MPI_File_write(mpi_flog, message, 28, MPI_CHAR, &mpi_status);           // 29: with '\0'
        } else {
            printf("shot HUNGRY, process %d\n", proc_rank);
            snprintf(message, 29, "shot HUNGRY, process %6d\n", proc_rank);
            MPI_File_seek(mpi_flog, 28 * (ishot - 1), MPI_SEEK_SET);
            MPI_File_write(mpi_flog, message, 28, MPI_CHAR, &mpi_status);
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
            float xmax=l*dt*velmax;
            int nleft=ncx_shot-xmax/unit-10;
            int nright=ncx_shot+xmax/unit+10;
            int nfront=ncy_shot-xmax/unit-10;
            int nback=ncy_shot+xmax/unit+10;
            int ntop=ncz_shot-xmax/unit-10;
            int nbottom=ncz_shot+xmax/unit+10;
            if(nleft<5) nleft=5;
            if(nright>nx-5) nright=nx-5;
            if(nfront<5) nfront=5;
            if(nback>ny-5) nback=ny-5;
            if(ntop<5) ntop=5;
            if(nbottom>nz-5) nbottom=nz-5;
            ntop = ntop-1;
            nfront = nfront-1;
            nleft = nleft-1;

            // Cmt out
            int ntmp = ntop;
            // Although up, vp, wp, us, vs, ws are modified below, we're sure there's no race condition.
            // Each loop accesses a UNIQUE element in the array, and the value is not used, no need to worry about the dirty cache
#pragma omp parallel for shared(u) shared(v) shared(w) shared(up1) shared(up2) shared(vp1) shared(vp2) shared(wp1) \
    shared(wp2) shared(us) shared(us1) shared(us2) shared(vs) shared(vs1) shared(vs2) shared(ws) shared(ws1) shared(ws2) \
    shared(vss) shared(vpp) shared(dr1) shared(dr2) shared(dtz) shared(dtx) shared(ncx_shot) shared(ncy_shot) shared(ncz_shot) \
    shared(wave)
            for(int k=ntop;k<nbottom;k++) {
                for(int j=nfront;j<nback;j++) {
                    for(int i=nleft;i<nright;i++)
                    {
                        float vvp2,drd1,drd2,vvs2;
                        float px,sx = 0.;
                        if(i==ncx_shot-1&&j==ncy_shot-1&&k==ncz_shot-1)
                        {
                            px=1.;
                        }
                        else
                        {
                            px=0.;
                        }
                        vvp2=vpp[k*ny*nx+j*nx+i]*vpp[k*ny*nx+j*nx+i];
                        drd1=dr1*vvp2;
                        drd2=dr2*vvp2;

                        vvs2=vss[k*ny*nx+j*nx+i]*vss[k*ny*nx+j*nx+i];
                        drd1=dr1*vvs2;
                        drd2=dr2*vvs2;

                        float tempux2 = 0;
                        float tempuy2 = 0;
                        float tempuz2 = 0;
                        float tempvx2 = 0;
                        float tempvy2 = 0;
                        float tempvz2 = 0;
                        float tempwx2 = 0;
                        float tempwy2 = 0;
                        float tempwz2 = 0;
                        float tempuxz = 0;
                        float tempuxy = 0;
                        float tempvyz = 0;
                        float tempvxy = 0;
                        float tempwxz = 0;
                        float tempwyz = 0;

                        for(int kk=1;kk<=mm;kk++) {
                            tempux2 += c_col0[kk-1]*(u[k*ny*nx+j*nx+(i+kk)]+u[k*ny*nx+j*nx+(i-kk)]);
                            tempuy2 += c_col0[kk-1]*(u[k*ny*nx+(j+kk)*nx+i]+u[k*ny*nx+(j-kk)*nx+i]);
                            tempuz2 += c_col0[kk-1]*(u[(k+kk)*ny*nx+j*nx+i]+u[(k-kk)*ny*nx+j*nx+i]);

                            tempvx2 += c_col0[kk-1]*(v[k*ny*nx+j*nx+(i+kk)]+v[k*ny*nx+j*nx+(i-kk)]);
                            tempvy2 += c_col0[kk-1]*(v[k*ny*nx+(j+kk)*nx+i]+v[k*ny*nx+(j-kk)*nx+i]);
                            tempvz2 += c_col0[kk-1]*(v[(k+kk)*ny*nx+j*nx+i]+v[(k-kk)*ny*nx+j*nx+i]);

                            tempwx2 += c_col0[kk-1]*(w[k*ny*nx+j*nx+(i+kk)]+w[k*ny*nx+j*nx+(i-kk)]);
                            tempwy2 += c_col0[kk-1]*(w[k*ny*nx+(j+kk)*nx+i]+w[k*ny*nx+(j-kk)*nx+i]);
                            tempwz2 += c_col0[kk-1]*(w[(k+kk)*ny*nx+j*nx+i]+w[(k-kk)*ny*nx+j*nx+i]);
                        }

                        tempux2=(tempux2+c0*u[k*ny*nx+j*nx+i])*vvp2*dtx*dtx;
                        tempuy2=(tempuy2+c0*u[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempuz2=(tempuz2+c0*u[k*ny*nx+j*nx+i])*vvs2*dtz*dtz;

                        tempvx2=(tempvx2+c0*v[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempvy2=(tempvy2+c0*v[k*ny*nx+j*nx+i])*vvp2*dtx*dtx;
                        tempvz2=(tempvz2+c0*v[k*ny*nx+j*nx+i])*vvs2*dtz*dtz;

                        tempwx2=(tempwx2+c0*w[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempwy2=(tempwy2+c0*w[k*ny*nx+j*nx+i])*vvs2*dtx*dtx;
                        tempwz2=(tempwz2+c0*w[k*ny*nx+j*nx+i])*vvp2*dtz*dtz;

                        // This loop is auto-vectorized
                        for(int kkk=1;kkk<=mm;kkk++)
                        {
                            for(int kk=1;kk<=mm;kk++)
                            {
                                tempuxz+=c[kkk-1][1+kk]*(
                                            +u[(k+kkk)*ny*nx+j*nx+(i+kk)]
                                            -u[(k-kkk)*ny*nx+j*nx+(i+kk)]
                                            +u[(k-kkk)*ny*nx+j*nx+(i-kk)]
                                            -u[(k+kkk)*ny*nx+j*nx+(i-kk)]);
                                // u[k+kkk][j][i+kk], u[k-kkk][j][i+kk], u[k-kkk][j][i-kk], u[k+kkk][j][i-kk]
                                tempuxy+=c[kkk-1][1+kk]*(
                                            +u[k*ny*nx+(j+kkk)*nx+(i+kk)]
                                            -u[k*ny*nx+(j-kkk)*nx+(i+kk)]
                                            +u[k*ny*nx+(j-kkk)*nx+(i-kk)]
                                            -u[k*ny*nx+(j+kkk)*nx+(i-kk)]);

                                tempvyz+=c[kkk-1][1+kk]*(
                                            +v[(k+kkk)*ny*nx+(j+kk)*nx+i]
                                            -v[(k-kkk)*ny*nx+(j+kk)*nx+i]
                                            +v[(k-kkk)*ny*nx+(j-kk)*nx+i]
                                            -v[(k+kkk)*ny*nx+(j-kk)*nx+i]);
                                tempvxy+=c[kkk-1][1+kk]*(
                                            +v[k*ny*nx+(j+kkk)*nx+(i+kk)]
                                            -v[k*ny*nx+(j-kkk)*nx+(i+kk)]
                                            +v[k*ny*nx+(j-kkk)*nx+(i-kk)]
                                            -v[k*ny*nx+(j+kkk)*nx+(i-kk)]);

                                tempwyz+=c[kkk-1][1+kk]*(
                                            +w[(k+kkk)*ny*nx+(j+kk)*nx+i]
                                            -w[(k-kkk)*ny*nx+(j+kk)*nx+i]
                                            +w[(k-kkk)*ny*nx+(j-kk)*nx+i]
                                            -w[(k+kkk)*ny*nx+(j-kk)*nx+i]);
                                tempwxz+=c[kkk-1][1+kk]*(
                                            +w[(k+kkk)*ny*nx+j*nx+(i+kk)]
                                            -w[(k-kkk)*ny*nx+j*nx+(i+kk)]
                                            +w[(k-kkk)*ny*nx+j*nx+(i-kk)]
                                            -w[(k+kkk)*ny*nx+j*nx+(i-kk)]);
                            } // for(kkk=1;kkk<=mm;kkk++) end
                        } //for(kk=1;kk<=mm;kk++) end

                        // LValues below are only changed here
                        up[k*ny*nx+j*nx+i]=2.*up1[k*ny*nx+j*nx+i]-up2[k*ny*nx+j*nx+i]
                                +tempux2+tempwxz*vvp2*dtz*dtx
                                +tempvxy*vvp2*dtz*dtx;
                        // up1[k][j][j], up2[k][j][i], up[k][j][i]
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
                }
            }

            // Again, those are UNIQUE access. Safe to share
#pragma omp parallel for shared(up) shared(us) shared(vp) shared(vs) shared(wp) shared(ws) shared(u) shared(v) shared(w) \
    shared(up2) shared(up1) shared(us2) shared(us1) shared(vp2) shared(vp1) shared(wp2) shared(wp1) shared(ws2) shared(ws1)
                for(int k=ntop;k<nbottom;k++) {
                    int vec_len = nright - nleft;
                    for(int j=nfront;j<nback;j++) {
//                        for(int i=nleft;i<nright;i++)
//                        {
//                            u[k*ny*nx+j*nx+i]=up[k*ny*nx+j*nx+i]+us[k*ny*nx+j*nx+i];
//                            v[k*ny*nx+j*nx+i]=vp[k*ny*nx+j*nx+i]+vs[k*ny*nx+j*nx+i];
//                            w[k*ny*nx+j*nx+i]=wp[k*ny*nx+j*nx+i]+ws[k*ny*nx+j*nx+i];

//                            up2[k*ny*nx+j*nx+i]=up1[k*ny*nx+j*nx+i];
//                            up1[k*ny*nx+j*nx+i]=up[k*ny*nx+j*nx+i];
//                            us2[k*ny*nx+j*nx+i]=us1[k*ny*nx+j*nx+i];
//                            us1[k*ny*nx+j*nx+i]=us[k*ny*nx+j*nx+i];
//                            vp2[k*ny*nx+j*nx+i]=vp1[k*ny*nx+j*nx+i];
//                            vp1[k*ny*nx+j*nx+i]=vp[k*ny*nx+j*nx+i];
//                            vs2[k*ny*nx+j*nx+i]=vs1[k*ny*nx+j*nx+i];
//                            vs1[k*ny*nx+j*nx+i]=vs[k*ny*nx+j*nx+i];
//                            wp2[k*ny*nx+j*nx+i]=wp1[k*ny*nx+j*nx+i];
//                            wp1[k*ny*nx+j*nx+i]=wp[k*ny*nx+j*nx+i];
//                            ws2[k*ny*nx+j*nx+i]=ws1[k*ny*nx+j*nx+i];
//                            ws1[k*ny*nx+j*nx+i]=ws[k*ny*nx+j*nx+i];
//                        }//for(i=nleft;i<nright;i++) end

                        int ary_position = k*ny*nx+j*nx+nleft;

                        vsAdd   (vec_len,   up + ary_position,  us + ary_position, u + ary_position);
                        vsAdd   (vec_len,   vp + ary_position,  vs + ary_position, v + ary_position);
                        vsAdd   (vec_len,   wp + ary_position,  ws + ary_position, w + ary_position);

                        cblas_scopy(vec_len, up1 + ary_position, 1, up2 + ary_position, 1);
                        cblas_scopy(vec_len, up + ary_position, 1,  up1 + ary_position, 1);
                        cblas_scopy(vec_len, vp1 + ary_position, 1, vp2 + ary_position, 1);
                        cblas_scopy(vec_len, vp + ary_position, 1,  vp1 + ary_position, 1);
                        cblas_scopy(vec_len, wp1 + ary_position, 1, wp2 + ary_position, 1);
                        cblas_scopy(vec_len, wp + ary_position, 1,  wp1 + ary_position, 1);
                        cblas_scopy(vec_len, us1 + ary_position, 1, us2 + ary_position, 1);
                        cblas_scopy(vec_len, us + ary_position, 1,  us1 + ary_position, 1);
                        cblas_scopy(vec_len, vs1 + ary_position, 1, vs2 + ary_position, 1);
                        cblas_scopy(vec_len, vs + ary_position, 1,  vs1 + ary_position, 1);
                        cblas_scopy(vec_len, ws1 + ary_position, 1, ws2 + ary_position, 1);
                        cblas_scopy(vec_len, ws + ary_position, 1,  ws1 + ary_position, 1);

                    }
                }


        }//for(l=1;l<=lt;l++) end
        // [Afa] Do we need to keep the order of data?
        // [Afa Update] Yes, we do need to KEEP THE ORDER of data
        //        fwrite(up+169*ny*nx,sizeof(float),ny*nx,fout);    // This is the original fwrite

        MPI_File_seek(mpi_fout, (ishot - 1) * ny * nx * sizeof(float), MPI_SEEK_SET);
        MPI_File_write(mpi_fout, up + 169 * ny * nx, ny * nx, MPI_FLOAT, &mpi_status);

    }//for(ishot=1;ishot<=nshot;ishot++) end

    MPI_File_close(&mpi_fout);
    MPI_File_close(&mpi_flog);

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
    return 0;
}

void zero_matrices(float *u, float *w, float *ws2, float *up2, float *vp1, float *wp1, float *us, float *ws, float *wp,
                   float *us2, float *us1, float *wp2, float *v, float *up1, int nz, int nx, float *up,
                   int ny, float *ws1, float *vs, float *vp2, float *vs1, float *vs2, float *vp)
{
    int t = nx * ny * nz;       // [Afa] Total elements number in an array
    // [Afa] That freaking big loop! Really bad for cache and SIMD. Decomposed it
    // AVX can process 8 float at a time

    float *matrices[21];

    matrices[0] = u;
    matrices[1] = w;
    matrices[2] = ws2;
    matrices[3] = up2;
    matrices[4] = vp1;
    matrices[5] = wp1;
    matrices[6] = us;
    matrices[7] = ws;
    matrices[8] = wp;
    matrices[9] = us2;
    matrices[10] = us1;
    matrices[11] = wp2;
    matrices[12] = v;
    matrices[13] = up1;
    matrices[14] = up;
    matrices[15] = ws1;
    matrices[16] = vs;
    matrices[17] = vp2;
    matrices[18] = vs1;
    matrices[19] = vs2;
    matrices[20] = vp;


    // After a bunch of profiling, this is the fastest way to init the array
    // In a loop of 20, this omp loop is 6 seconds faster than sequential execution,
    // seq exec is 9 seconds faster than omp sections
#ifdef __STDC_IEC_559__
#pragma omp parallel for shared(matrices)
    for (int i = 0; i < 21; ++i)
        memset(matrices[i], 0, sizeof(float) * t);
#else
    #pragma omp parallel for
    for (int i = 0; i < 21; ++i) {
        #pragma omp simd collapse(8)
        for (int j = 0; j < t; ++j)
            matrices[i][j] = 0.0f;
    }
#endif
}









