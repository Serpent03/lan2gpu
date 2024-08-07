c_dll_out = comp.dll
c_dll_out_path = build/

cc = cl
bash = bash -c
ccflags = /D_USRDLL /D_WINDLL
ccflags2 = /MT /link /DLL
c_dll_source = src/computation.c

cudacc = nvcc
cuda_dll_source = src/cudaimpl.cu
cuda_libs = lib/*
cuda_obj = cuda.obj

run:
	flask --app server run --host=0.0.0.0

dll: $(c_dll_out) 

$(c_dll_out): $(c_dll_source) $(cuda_obj) 
	$(cc) $(ccflags) $^ $(cuda_libs) $(ccflags2) /OUT:$(c_dll_out_path)$@

$(cuda_obj): $(cuda_dll_source)
	$(cudacc) -c -o $@ $^

clean:
	del *.exe
	del *.obj
	del *.o
	del .\build\*.dll
	del .\build\*.lib
	del .\build\*.exp