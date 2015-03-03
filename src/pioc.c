/**
 * @file pioc.c
 * @author Jim Edwards
 * @date  2014
 * @brief PIO C interface 
 *
 * 
 * 
 * 
 * @see http://code.google.com/p/parallelio/
 */


#include <pio.h>
#include <pio_internal.h>


static int counter=0;
static bool PIO_Save_Decomps=false;
/**
 ** @brief Check to see if PIO has been initialized.
 */
int PIOc_iosystem_is_active(const int iosysid, bool *active)
{
  iosystem_desc_t *ios;
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;
  
  if(ios->comp_comm == MPI_COMM_NULL && ios->io_comm == MPI_COMM_NULL){
    *active = false;
  }else{
    *active = true;
  }
  return PIO_NOERR;
}
/**
 ** @brief Check to see if PIO file is open.
 */

int PIOc_File_is_Open(int ncid)
{
  file_desc_t *file;
  file = pio_get_file_from_id(ncid);
  if(file==NULL)
    return 0;
  else
    return 1;
}

/**
 ** @brief Set the error handling method to be used for subsequent 
 ** pio library calls, returns the previous method setting
 */
int PIOc_Set_File_Error_Handling(int ncid, int method)
{
  file_desc_t *file;
  int oldmethod;
  file = pio_get_file_from_id(ncid);
  oldmethod = file->iosystem->error_handler;
  file->iosystem->error_handler = method;
  return(oldmethod);
} 

/**
 ** @brief Increment the unlimited dimension of the given variable
 */
int PIOc_advanceframe(int ncid, int varid)
{
  file_desc_t *file;
  file = pio_get_file_from_id(ncid);
  if(file == NULL)
    return PIO_EBADID;

  file->varlist[varid].record++;

  return(PIO_NOERR);
} 

/**
 ** @brief Set the unlimited dimension of the given variable
 */
int PIOc_setframe(const int ncid, const int varid,const int frame)
{
  file_desc_t *file;
  file = pio_get_file_from_id(ncid);
  if(file == NULL || varid<0 || varid>=PIO_MAX_VARS){
    return PIO_EBADID;
  }

  file->varlist[varid].record = frame;

  return(PIO_NOERR);
} 

/**
 ** @brief Get the number of IO tasks set.
 */
int PIOc_get_numiotasks(int iosysid, int *numiotasks)
{
  iosystem_desc_t *ios;
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;

  *numiotasks = ios->num_iotasks;

  return PIO_NOERR;

}


/**
 ** @brief Get the IO rank on the current task
 */
int PIOc_get_iorank(int iosysid, int *iorank)
{
  iosystem_desc_t *ios;
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;

  *iorank = ios->io_rank;

  return PIO_NOERR;

}

/**
 ** @brief Get the local size of the variable
 */

int PIOc_get_local_array_size(int ioid)
{
  io_desc_t *iodesc;
  iodesc = pio_get_iodesc_from_id(ioid);
  return(iodesc->ndof);
}

/**
 ** @ingroup PIO_error_method
 ** @brief Set the error handling method used for subsequent calls
 */

 int PIOc_Set_IOSystem_Error_Handling(int iosysid, int method)
{
  iosystem_desc_t *ios;
  int oldmethod;
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios==NULL){
    fprintf(stderr,"%s %d Error setting eh method\n",__FILE__,__LINE__);
    print_trace(stderr);
    return PIO_EBADID;
  }	
  oldmethod = ios->error_handler;
  ios->error_handler = method;
  return(oldmethod);
}  

/**
 ** @ingroup PIO_initdecomp
 ** @brief C interface to the initdecomp
 ** @param  iosysid @copydoc iosystem_desc_t (input)
 ** @param  basetype the basic PIO data type used (input)
 ** @param  ndims the number of dimensions in the variable (input)
 ** @param  dims[] the global size of each dimension (input)
 ** @param  maplen the local length of the compmap array (input)
 ** @param  compmap[] a 1 based array of offsets into the array record on file.  A 0 in this array indicates a value which should not be transfered. (input)
 ** @param ioidp  the io description pointer (output)
 ** @param rearranger the rearranger to be used for this decomp or NULL to use the default (optional input)
 ** @param iostart An optional array of start values for block cyclic decompositions  (optional input)
 ** @param iocount An optional array of count values for block cyclic decompositions  (optional input)
 */


int PIOc_InitDecomp(const int iosysid, const int basetype,const int ndims, const int dims[], 
		    const int maplen, const PIO_Offset *compmap, int *ioidp,const int *rearranger,  
		    const PIO_Offset *iostart,const PIO_Offset *iocount)
{
  iosystem_desc_t *ios;
  io_desc_t *iodesc;
  int mpierr;
  int ierr;
  int iosize;
  int ndisp;



  for(int i=0;i<ndims;i++){
    if(dims[i]<=0){
      piodie("Invalid dims argument",__FILE__,__LINE__);
    }
  }
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;
  

  if(PIO_Save_Decomps){
    char filename[30];
    if(ios->num_comptasks < 100) {
      sprintf(filename, "piodecomp%2.2dtasks%2.2ddims%2.2d.dat",ios->num_comptasks,ndims,counter);
    }else if(ios->num_comptasks < 10000) {
      sprintf(filename, "piodecomp%4.4dtasks%2.2ddims%2.2d.dat",ios->num_comptasks,ndims,counter);
    }else{
      sprintf(filename, "piodecomp%6.6dtasks%2.2ddims%2.2d.dat",ios->num_comptasks,ndims,counter);
    }
    PIOc_writemap(filename,ndims,dims,maplen,compmap,ios->comp_comm);
    counter++;
  }


  iodesc = malloc_iodesc(basetype, ndims);
  if(rearranger == NULL)
    iodesc->rearranger = ios->default_rearranger;
  else
    iodesc->rearranger = *rearranger;
    
  if(iodesc->rearranger==PIO_REARR_SUBSET){
    if((iostart != NULL) && (iocount != NULL)){ 
      fprintf(stderr,"%s %s\n","Iostart and iocount arguments to PIOc_InitDecomp",
	      "are incompatable with subset rearrange method and will be ignored");
    }
    iodesc->num_aiotasks = ios->num_iotasks;
    ierr = subset_rearrange_create( *ios, maplen, compmap, dims, ndims, iodesc);
  }else{   
      if(ios->ioproc){
      //  Unless the user specifies the start and count for each IO task compute it.    
	if((iostart != NULL) && (iocount != NULL)){ 
	  //	  printf("iocount[0] = %ld %ld\n",iocount[0], iocount);
	  iodesc->maxiobuflen=1;
	  for(int i=0;i<ndims;i++){
	    iodesc->firstregion->start[i] = iostart[i];
	    iodesc->firstregion->count[i] = iocount[i];
	    compute_maxIObuffersize(ios->io_comm, iodesc);
	    
	  }
	  iodesc->num_aiotasks = ios->num_iotasks;
	}else{
	  iodesc->num_aiotasks = CalcStartandCount(basetype, ndims, dims, 
						   ios->num_iotasks, ios->io_rank,
						   iodesc->firstregion->start, iodesc->firstregion->count);
	  if(ios->iomaster){
	    printf("%s %d aiotasks=%d\n",__FILE__,__LINE__,iodesc->num_aiotasks);
	  }
      }
      compute_maxIObuffersize(ios->io_comm, iodesc);

    }
    // Depending on array size and io-blocksize the actual number of io tasks used may vary
    CheckMPIReturn(MPI_Bcast(&(iodesc->num_aiotasks), 1, MPI_INT, ios->ioroot,
			     ios->my_comm),__FILE__,__LINE__);
    // Compute the communications pattern for this decomposition
    if(iodesc->rearranger==PIO_REARR_BOX){   
      ierr = box_rearrange_create( *ios, maplen, compmap, dims, ndims, iodesc);
    }
    /*
    if(ios->ioproc){
      io_region *ioregion = iodesc->firstregion;
      while(ioregion != NULL){
	for(int i=0;i<ndims;i++)
	  printf("%s %d i %d dim %d start %ld count %ld\n",__FILE__,__LINE__,i,dims[i],ioregion->start[i],ioregion->count[i]);
	ioregion = ioregion->next;
      }
    }
    */
  }

  *ioidp = pio_add_to_iodesc_list(iodesc);
  
  return PIO_NOERR;
}

/**
 ** @ingroup PIO_initdecomp
 ** This is a simplified initdecomp which can be used if the memory order of the data can be 
 ** expressed in terms of start and count on the file.
 ** in this case we compute the compdof and use the subset rearranger
 */


int PIOc_InitDecomp_bc(const int iosysid, const int basetype,const int ndims, const int dims[], 
		       const long int start[], const long int count[], int *ioidp)
		    
{
  iosystem_desc_t *ios;
  io_desc_t *iodesc;
  int mpierr;
  int ierr;
  int iosize;
  int ndisp;

  
  for(int i=0;i<ndims;i++){
    if(dims[i]<=0){
      piodie("Invalid dims argument",__FILE__,__LINE__);
    }
    if(start[i]<0 || count[i]< 0 || (start[i]+count[i])>dims[i]){
      piodie("Invalid start or count argument ",__FILE__,__LINE__);
    }
  }
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;

  int n, i, maplen=1;
    
  for( i=0;i<ndims;i++){
    maplen*=count[i];
  }
  PIO_Offset compmap[maplen], prod[ndims], loc[ndims];
    
  prod[ndims-1]=1;
  loc[ndims-1]=0;
  for(n=ndims-2;n>=0;n--){
    prod[n]=prod[n+1]*dims[n+1];  
    loc[n]=0;
  }
  for(i=0;i<maplen;i++){
    compmap[i]=0;
    for(n=ndims-1;n>=0;n--){
      compmap[i]+=(start[n]+loc[n])*prod[n];
    }
    n=ndims-1;
    loc[n]=(loc[n]+1)%count[n];
    while(loc[n]==0 && n>0){
      n--;
      loc[n]=(loc[n]+1)%count[n];
    }
  }
  int rearr = PIO_REARR_SUBSET;
  PIOc_InitDecomp( iosysid, basetype,ndims, dims, 
		   maplen,  compmap, ioidp, &rearr, NULL, NULL);  


  return PIO_NOERR;
}

/** 
 ** @ingroup PIO_init
 ** @brief library initialization used when IO tasks are a subset of compute tasks
 ** @param comp_comm the MPI_Comm of the compute tasks
 ** @param num_iotasks the number of io tasks to use
 ** @param stride the offset between io tasks in the comp_comm
 ** @param base the comp_comm index of the first io task
 ** @param rearr the rearranger to use by default, this may be overriden in the @ref PIO_initdecomp
 ** @param iosysidp index of the defined system descriptor
 */

int PIOc_Init_Intracomm(const MPI_Comm comp_comm, 
			const int num_iotasks, const int stride, 
			const int base,const int rearr, int *iosysidp)
{
  iosystem_desc_t *iosys;
  int ierr;
  MPI_Group compgroup, iogroup;
  int lbase;
  iosys = (iosystem_desc_t *) malloc(sizeof(iosystem_desc_t));

  iosys->union_comm = comp_comm;
  iosys->comp_comm = comp_comm;
  iosys->my_comm = comp_comm;
  iosys->io_comm = MPI_COMM_NULL;
  iosys->intercomm = MPI_COMM_NULL;
  iosys->error_handler = PIO_INTERNAL_ERROR;
  iosys->async_interface= false;
  iosys->compmaster = false;
  iosys->iomaster = false;
  iosys->ioproc = false;
  iosys->default_rearranger = rearr;
  iosys->num_iotasks = num_iotasks;

  CheckMPIReturn(MPI_Comm_rank(comp_comm, &(iosys->comp_rank)),__FILE__,__LINE__);
  CheckMPIReturn(MPI_Comm_size(comp_comm, &(iosys->num_comptasks)),__FILE__,__LINE__);
  if(iosys->comp_rank==0)
    iosys->compmaster = true;  
#ifdef BGQxxx
  lbase = base;
  determineiotasks(comp_comm, &(iosys->num_iotasks), &lbase, &stride, &rearr, &(iosys->ioproc));
  if(iosys->comp_rank==0)
    printf("%s %d %d\n",__FILE__,__LINE__,iosys->num_iotasks);
  if(iosys->ioproc)
    printf("%s %d %d\n",__FILE__,__LINE__,iosys->comp_rank);
    
#else
  if((num_iotasks < 1) || ((num_iotasks*stride) > iosys->num_comptasks)){
    return PIO_EBADID;
  }
  iosys->ioranks = (int *) calloc(sizeof(int), iosys->num_iotasks);
  for(int i=0;i< num_iotasks; i++){
    iosys->ioranks[i] = (base + i*stride) % iosys->num_comptasks;
    if(iosys->ioranks[i] == iosys->comp_rank)
      iosys->ioproc = true;
  }
  iosys->ioroot = iosys->ioranks[0];
#endif

#ifndef _MPISERIAL
  CheckMPIReturn(MPI_Info_create(&(iosys->info)),__FILE__,__LINE__);
  iosys->info = MPI_INFO_NULL;
#endif


  if(iosys->comp_rank == iosys->ioranks[0])
    iosys->iomaster = true;

  CheckMPIReturn(MPI_Comm_group(comp_comm, &compgroup),__FILE__,__LINE__);
			
  CheckMPIReturn(MPI_Group_incl(compgroup, num_iotasks, iosys->ioranks, &iogroup),__FILE__,__LINE__);

  CheckMPIReturn(MPI_Comm_create(comp_comm, iogroup, &(iosys->io_comm)),__FILE__,__LINE__);
  if(iosys->ioproc)
    CheckMPIReturn(MPI_Comm_rank(iosys->io_comm, &(iosys->io_rank)),__FILE__,__LINE__);
  else
    iosys->io_rank = -1;

  iosys->union_rank = iosys->comp_rank;
  
  *iosysidp = pio_add_to_iosystem_list(iosys);
  char *envptr = getenv("PIO_Save_Decomps");
  //  printf("%s %d %s\n",__FILE__,__LINE__,envptr);
  if(envptr != NULL && (strcmp(envptr,"true")==0)){
    PIO_Save_Decomps=true;
  }
  /* allocate buffer space for compute nodes */
  compute_buffer_init(*iosys);

  return PIO_NOERR;
}

/**
 ** @internal 
 ** interface to call from pio_init from fortran
 ** @endinternal
 */
int PIOc_Init_Intracomm_from_F90(int f90_comp_comm, 
			const int num_iotasks, const int stride, 
				 const int base, const int rearr, int *iosysidp){
  return PIOc_Init_Intracomm(MPI_Comm_f2c(f90_comp_comm), num_iotasks, stride,base,rearr, iosysidp);
}
  
/**
 ** @brief Send a hint to the MPI-IO library 
 **
 */
int PIOc_set_hint(const int iosysid, char hint[], const char hintval[])
{
  iosystem_desc_t *ios;

  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;
#ifndef _MPISERIAL  
  if(ios->ioproc)
    CheckMPIReturn( MPI_Info_set(ios->info, hint, hintval), __FILE__,__LINE__);
#endif
  return PIO_NOERR;

}

/**
 ** @brief Clean up data structures and exit the pio library
 */

int PIOc_finalize(const int iosysid)
{
  iosystem_desc_t *ios, *nios;

  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID; 
  /* FIXME: The memory for ioranks is allocated in C only for intracomms
   * Remove this check once mem allocs for ioranks completely moves to the
   * C code
   */ 
  if(ios->intercomm == MPI_COMM_NULL){
    if(ios->ioranks != NULL){
      free(ios->ioranks);
    }
  }

  free_cn_buffer_pool(*ios);


  if(ios->io_comm != MPI_COMM_NULL){
    MPI_Comm_free(&(ios->io_comm));
  }
  return pio_delete_iosystem_from_list(iosysid);

  
}

/**
 ** @brief return a logical indicating whether this task is an iotask 
 */
int PIOc_iam_iotask(const int iosysid, bool *ioproc)
{
  iosystem_desc_t *ios;
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;
  
  *ioproc = ios->ioproc;
  return PIO_NOERR;
}

/**
 ** @brief return the rank of this task in the io comm or
 ** -1 if this task is not in the comm
 */
int PIOc_iotask_rank(const int iosysid, int *iorank)
{
  iosystem_desc_t *ios;
  ios = pio_get_iosystem_from_id(iosysid);
  if(ios == NULL)
    return PIO_EBADID;

  *iorank = ios->io_rank;

  return PIO_NOERR;
  
}
