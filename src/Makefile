#
#KERNEL_CPU_OPT=1
KERNEL_CPU_REF=1
KERNEL_DPC=1
OPENMP=1
#
RESULT_CHECK=1
#REPVEC=1

DPCPP_CXX= dpcpp
DPCPP_CXXFLAGS= -O3 -std=c++20 -fmax-errors=3 
DPCPP_CXXFLAGS+= -qmkl
DPCPP_CXXFLAGS+= -DDP # -DDP for double precision
#DPCPP_CXXFLAGS+= -DREPEAT=4 # Number of repetitions + 1 warm up

#DPCPP_CXXFLAGS+= -DDPC_BLOCK_D1=2 -DDPC_BLOCK_D2=2 -DDPC_BLOCK_D3=16
#DPCPP_CXXFLAGS+= -DCPU_BLOCK_D1=4 -DCPU_BLOCK_D2=8 -DCPU_BLOCK_D3=16 -DCPU_BLOCK_X=32

ifdef KERNEL_CPU_REF
	DPCPP_CXXFLAGS+= -DKERNEL_CPU_REF -DOMP
endif
ifdef KERNEL_CPU_OPT
	DPCPP_CXXFLAGS+= -DKERNEL_CPU_OPT -DOMP
endif
ifdef KERNEL_DPC
	DPCPP_CXXFLAGS+= -DKERNEL_DPC
endif
ifdef RESULT_CHECK
	DPCPP_CXXFLAGS+= -DRESULT_CHECK
endif

ifdef OPENMP
	DPCPP_CXXFLAGS+= -qopenmp
endif
ifdef REPVEC
	DPCPP_CXXFLAGS+= -qopt-report=5 -qopt-report-phase=vec,loop
endif

#
SOURCES=lattice.cc misc.cc kernel_dpcpp.cc kernel_cpu_ref.cc kernel_cpu_opt.cc main.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=kernel

#
%.o: %.cc
	$(DPCPP_CXX) -c $(DPCPP_CXXFLAGS) $< -o $@ $(DEFINES)

#
all: $(OBJECTS)
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(LDFLAGS) $(OBJECTS) -o $(EXECUTABLE)

#
clean:
	rm -f *.o $(EXECUTABLE) *.optrpt 
