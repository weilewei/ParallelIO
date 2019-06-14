/*
 * Tests the PIO library with multiple iosysids in use at the
 * same time.
 *
 * This is a simplified, C version of the fortran pio_iosystem_tests2.F90.
 *
 * Ed Hartnett
 */
#include <config.h>
#include <pio.h>
#include <pio_tests.h>

/* The number of tasks this test should run on. */
#define TARGET_NTASKS 4

/* The name of this test. */
#define TEST_NAME "test_z5_create_file"

/* Used to define netcdf test file. */
#define PIO_TF_MAX_STR_LEN 100
#define ATTNAME "var0"
//#define DIMNAME "filename_dim"
//#define DIMNAME "pio_iosys_test_file0.z5/filename_dim"
/* This creates a netCDF file in the specified format, with some
 * sample values. */
int create_file(MPI_Comm comm, int iosysid, int format, char *filename,
                char *attname, char *dimname, int my_rank, int *ncidp)
{
    int ncid, varid, dimid0, dimid1;
    int ret;

    /* Create the file. */
    if ((ret = PIOc_createfile(iosysid, ncidp, &format, filename, NC_CLOBBER)))
        return ret;
    MPI_Barrier(comm);
//    printf("end of PIOc_createfile ncidp:%d\n", *ncidp);
//    /* Use the ncid to set the IO system error handler. This function
//     * is deprecated. */
//    PIOc_Set_File_Error_Handling(ncid, PIO_RETURN_ERROR);
//    int method = PIOc_Set_File_Error_Handling(ncid, PIO_RETURN_ERROR);
//    if (method != PIO_RETURN_ERROR)
//        return ERR_WRONG;
//
    /* Define a dimension. */
//    dim_desc_t *dim0;
    char dimname0[] = "lat";
    int dimval0 = 100;

    char dimname1[] = "lon";
    int dimval1 = 200;

    if ((ret = PIOc_def_dim(*ncidp, dimname0, dimval0, &dimid0)))
        return ret;

    if ((ret = PIOc_def_dim(*ncidp, dimname1, dimval1, &dimid1)))
        return ret;

    int twod_dimids[2];
    twod_dimids[0] = dimid0;
    twod_dimids[1] = dimid1;
    MPI_Barrier(comm);
    /* Define a 1-D variable. */
    if ((ret = PIOc_def_var(*ncidp, attname, NC_INT64, 2, &twod_dimids, &varid)))
        return ret;
    MPI_Barrier(comm);
    /* Write an attribute. */
    char attributename0[] = "time";
    char attributeval0[] = "noon";
    if ((ret = PIOc_put_att_text(*ncidp, varid, attributename0, strlen(filename), attributeval0)))
        return ret;

    char attributename1[] = "long";
    double attributeval1 = 42.0;
    if ((ret = PIOc_put_att_double(*ncidp, varid, attributename1, NC_INT64, 1, &attributeval1)))
        return ret;
    // TODO: Z5Z5 how to get number of ranks
    printf("I am rank: %d\n", my_rank);
    long int count[] = {100/4, 50};
    long int start[] = {my_rank*100/4, my_rank*200/4};
    long int int64_array[100][200];
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 200; j++)
        {
            int64_array[i][j]=42;
        }
    }
//    MPI_Comm test_comm;
    MPI_Barrier(comm);

    if ((ret = PIOc_put_vara_int(*ncidp, varid, start, count, (int *)int64_array)))
        ERR(ret);
    //PIOc_put_vars_tc

    /* End define mode. */
//    if ((ret = PIOc_enddef(ncid)))
//        return ret;
//    printf("end of PIOc_enddef ret:%d\n", ret);
    /* Close the file. */
//    if ((ret = PIOc_closefile(ncid)))
//        return ret;
    return PIO_NOERR;
}

///* This checks an already-open netCDF file. */
//int check_file(MPI_Comm comm, int iosysid, int format, int ncid, char *filename,
//               char *attname, char *dimname, int my_rank)
//{
//    int dimid;
//    int ret;
//
//    /* Check the file. */
//    if ((ret = PIOc_inq_dimid(ncid, dimname, &dimid)))
//        return ret;
//
//    return PIO_NOERR;
//}
//
///* This opens and checks a netCDF file. */
//int open_and_check_file(MPI_Comm comm, int iosysid, int iotype, int *ncid, char *fname,
//                        char *attname, char *dimname, int disable_close, int my_rank)
//{
//    int mode = PIO_WRITE;
//    int ret;
//
//    /* Open the file. */
//    if ((ret = PIOc_openfile(iosysid, ncid, &iotype, fname, mode)))
//        return ret;
//
//    /* Check the file. */
//    if ((ret = check_file(comm, iosysid, iotype, *ncid, fname, attname, dimname, my_rank)))
//        return ret;
//
//    /* Close the file, maybe. */
//    if (!disable_close)
//        if ((ret = PIOc_closefile(*ncid)))
//            return ret;
//
//    return PIO_NOERR;
//}

/* Run async tests. */
int main(int argc, char **argv)
{
    int my_rank; /* Zero-based rank of processor. */
    int ntasks; /* Number of processors involved in current execution. */
    int iosysid; /* The ID for the parallel I/O system. */
    int iosysid_world; /* The ID for the parallel I/O system. */
    int ret; /* Return code. */
    int num_flavors;
    int iotypes[NUM_FLAVORS];
    MPI_Comm test_comm;

    /* Initialize test. */
    if ((ret = pio_test_init2(argc, argv, &my_rank, &ntasks, TARGET_NTASKS, TARGET_NTASKS,
                              -1, &test_comm)))
        ERR(ERR_INIT);

    /* Test code runs on TARGET_NTASKS tasks. The left over tasks do
     * nothing. */
    if (my_rank < TARGET_NTASKS)
    {
        /* Figure out iotypes. */
        if ((ret = get_iotypes(&num_flavors, iotypes)))
            ERR(ret);

        /* Split world into odd and even. */
        MPI_Comm newcomm;
        int even = my_rank % 2 ? 0 : 1;
        if ((ret = MPI_Comm_split(test_comm, even, 0, &newcomm)))
            MPIERR(ret);

        /* Get rank in new communicator and its size. */
        int new_rank, new_size;
        if ((ret = MPI_Comm_rank(newcomm, &new_rank)))
            MPIERR(ret);
        if ((ret = MPI_Comm_size(newcomm, &new_size)))
            MPIERR(ret);

        /* Initialize PIO system. */
        if ((ret = PIOc_Init_Intracomm(newcomm, 2, 1, 0, 1, &iosysid)))
            ERR(ret);

        /* This should fail. */
        if (PIOc_finalize(iosysid + TEST_VAL_42) != PIO_EBADID)
            ERR(ERR_WRONG);

        /* Initialize another PIO system. */
        if ((ret = PIOc_Init_Intracomm(test_comm, 4, 1, 0, 1, &iosysid_world)))
            ERR(ret);

        int file0id;
        int file1id;
        int file2id;

        for (int i = 0; i < num_flavors; i++)
        {
            if (iotypes[i] == 5)
            {
                char fname0[] = "pio_iosys_test_file0";
                char fname1[] = "pio_iosys_test_file1";
                char fname2[] = "pio_iosys_test_file2";
                char* group_tmp = "/group";
                char* dimname = "dim1";
//                char* dimname = (char*) malloc (1 + strlen(fname0) + strlen(group_tmp) + strlen(dim_tmp));
//                strcpy(dimname, fname0);
//                strcat(dimname, group_tmp);
//                strcat(dimname, dim_tmp);
//                printf("dimname is %s\n", dimname);
                if ((ret = create_file(test_comm, iosysid_world, iotypes[i], fname0, ATTNAME,
                                       dimname, my_rank, &file0id)))
                    ERR(ret);
//                printf("1st ncid=%d\n", file0id);
//                if ((ret = create_file(test_comm, iosysid_world, iotypes[i], fname1, ATTNAME,
//                                       dimname, my_rank, &file1id)))
//                    ERR(ret);
//                printf("2nd ncid=%d\n", file1id);
//                if ((ret = create_file(test_comm, iosysid_world, iotypes[i], fname2, ATTNAME,
//                                       dimname, my_rank, &file2id)))
//                    ERR(ret);
//                printf("3rd ncid=%d\n", file2id);
                MPI_Barrier(test_comm);


//                if ((ret = PIOc_def_var(z5_file0.z5id, z5_var1.varname, PIO_FLOAT, 1, &dimid, &z5_var1.varid)))
//                    return ret;
//            /* Now check the first file. */
//            int ncid;
//            if ((ret = open_and_check_file(test_comm, iosysid_world, iotypes[i], &ncid, fname0,
//                                           ATTNAME, DIMNAME, 1, my_rank)))
//                ERR(ret);
//
//            /* Now have the odd/even communicators each check one of the
//             * remaining files. */
//            int ncid2;
//            char *fname = even ? fname1 : fname2;
//            if ((ret = open_and_check_file(newcomm, iosysid, iotypes[i], &ncid2, fname,
//                                           ATTNAME, DIMNAME, 1, my_rank)))
//                ERR(ret);
//
//
//            /* Close the still-open files. */
//            if ((ret = PIOc_closefile(ncid)))
//                ERR(ret);
//            if ((ret = PIOc_closefile(ncid2)))
//                ERR(ret);

                /* Wait for everyone to finish. */

                if ((ret = MPI_Barrier(test_comm)))
                    MPIERR(ret);
            }

        } /* next iotype */

        if ((ret = MPI_Comm_free(&newcomm)))
            MPIERR(ret);

        /* Finalize PIO system. */
        if ((ret = PIOc_finalize(iosysid)))
            ERR(ret);

        /* Finalize PIO system. */
        if ((ret = PIOc_finalize(iosysid_world)))
            ERR(ret);
    } /* my_rank < TARGET_NTASKS */

    /* Finalize test. */
    if ((ret = pio_test_finalize(&test_comm)))
        return ret;

    printf("%d %s SUCCESS!!\n", my_rank, TEST_NAME);

    return 0;
}
