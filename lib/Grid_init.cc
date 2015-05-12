/****************************************************************************/
/* pab: Signal magic. Processor state dump is x86-64 specific               */
/****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h> 
#include <sys/time.h>
#include <signal.h>
#include <iostream>
#include <Grid.h>
#include <algorithm>

#undef __X86_64
#define MAC

#ifdef MAC
#include <execinfo.h>
#endif

namespace Grid {

  //////////////////////////////////////////////////////
  // Convenience functions to access stadard command line arg
  // driven parallelism controls
  //////////////////////////////////////////////////////
  static std::vector<int> Grid_default_simd;
  static std::vector<int> Grid_default_latt;
  static std::vector<int> Grid_default_mpi;

  int GridThread::_threads;


  const std::vector<int> &GridDefaultSimd(void)     {return Grid_default_simd;};
  const std::vector<int> &GridDefaultLatt(void)     {return Grid_default_latt;};
  const std::vector<int> &GridDefaultMpi(void)      {return Grid_default_mpi;};


  ////////////////////////////////////////////////////////////
  // Command line parsing assist for stock controls
  ////////////////////////////////////////////////////////////
std::string GridCmdOptionPayload(char ** begin, char ** end, const std::string & option)
{
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    std::string payload(*itr);
    return payload;
  }
  return std::string("");
}
bool GridCmdOptionExists(char** begin, char** end, const std::string& option)
{
  return std::find(begin, end, option) != end;
}

void GridCmdOptionIntVector(std::string &str,std::vector<int> & vec)
{
  vec.resize(0);
  std::stringstream ss(str);
  int i;
  while (ss >> i){
    vec.push_back(i);
    if(std::ispunct(ss.peek()))
      ss.ignore();
  }    
  return;
}

void GridParseLayout(char **argv,int argc,
		     std::vector<int> &latt,
		     std::vector<int> &simd,
		     std::vector<int> &mpi)
{
  mpi =std::vector<int>({1,1,1,1});
  latt=std::vector<int>({8,8,8,8});

#if defined(SSE4)
  simd=std::vector<int>({1,1,1,2});
#endif
#if defined(AVX1) || defined (AVX2)
  simd=std::vector<int>({1,1,2,2});
#endif
#if defined(AVX512)
  simd=std::vector<int>({1,2,2,2});
#endif

  GridThread::SetMaxThreads();

  std::string arg;
  if( GridCmdOptionExists(argv,argv+argc,"--mpi") ){
    arg = GridCmdOptionPayload(argv,argv+argc,"--mpi");
    GridCmdOptionIntVector(arg,mpi);
  }
  if( GridCmdOptionExists(argv,argv+argc,"--simd") ){
    arg= GridCmdOptionPayload(argv,argv+argc,"--simd");
    GridCmdOptionIntVector(arg,simd);
  }
  if( GridCmdOptionExists(argv,argv+argc,"--grid") ){
    arg= GridCmdOptionPayload(argv,argv+argc,"--grid");
    GridCmdOptionIntVector(arg,latt);
  }
  if( GridCmdOptionExists(argv,argv+argc,"--omp") ){
    std::vector<int> ompthreads(0);
    arg= GridCmdOptionPayload(argv,argv+argc,"--omp");
    GridCmdOptionIntVector(arg,ompthreads);
    assert(ompthreads.size()==1);
    GridThread::SetThreads(ompthreads[0]);
  }

}

  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
void Grid_init(int *argc,char ***argv)
{
#ifdef GRID_COMMS_MPI
  MPI_Init(argc,argv);
#endif
  // Parse command line args.

  if( GridCmdOptionExists(*argv,*argv+*argc,"--debug-signals") ){
    Grid_debug_handler_init();
  }
  if( !GridCmdOptionExists(*argv,*argv+*argc,"--debug-stdout") ){
    Grid_quiesce_nodes();
  }
  GridParseLayout(*argv,*argc,
		  Grid_default_latt,
		  Grid_default_simd,
		  Grid_default_mpi);

}


  ////////////////////////////////////////////////////////////
  // Verbose limiter on MPI tasks
  ////////////////////////////////////////////////////////////
  void Grid_quiesce_nodes(void)
  {
#ifdef GRID_COMMS_MPI
    int me;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    if ( me ) { 
      std::cout.setstate(std::ios::badbit);
    }
#endif
  }
  void Grid_unquiesce_nodes(void)
  {
#ifdef GRID_COMMS_MPI
    std::cout.clear();
#endif
  }

  
void Grid_finalize(void)
{
#ifdef GRID_COMMS_MPI
  MPI_Finalize();
  Grid_unquiesce_nodes();
#endif
}
double usecond(void) {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1.0*tv.tv_usec + 1.0e6*tv.tv_sec;
}

#define _NBACKTRACE (256)
void * Grid_backtrace_buffer[_NBACKTRACE];

void Grid_sa_signal_handler(int sig,siginfo_t *si,void * ptr)
{
  printf("Caught signal %d\n",si->si_signo);
  printf("  mem address %llx\n",(unsigned long long)si->si_addr);
  printf("         code %d\n",si->si_code);

#ifdef __X86_64
    ucontext_t * uc= (ucontext_t *)ptr;
  struct sigcontext *sc = (struct sigcontext *)&uc->uc_mcontext;
  printf("  instruction %llx\n",(unsigned long long)sc->rip);
#define REG(A)  printf("  %s %lx\n",#A,sc-> A);
  REG(rdi);
  REG(rsi);
  REG(rbp);
  REG(rbx);
  REG(rdx);
  REG(rax);
  REG(rcx);
  REG(rsp);
  REG(rip);


  REG(r8);
  REG(r9);
  REG(r10);
  REG(r11);
  REG(r12);
  REG(r13);
  REG(r14);
  REG(r15);
#endif
#ifdef MAC
  int symbols    = backtrace        (Grid_backtrace_buffer,_NBACKTRACE);
  char **strings = backtrace_symbols(Grid_backtrace_buffer,symbols);
  for (int i = 0; i < symbols; i++){
    printf ("%s\n", strings[i]);
  }
#endif
  exit(0);
  return;
};

void Grid_debug_handler_init(void)
{
  struct sigaction sa,osa;
  sigemptyset (&sa.sa_mask);
  sa.sa_sigaction= Grid_sa_signal_handler;
  sa.sa_flags    = SA_SIGINFO;
  sigaction(SIGSEGV,&sa,NULL);
  sigaction(SIGTRAP,&sa,NULL);
}
}