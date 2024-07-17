c_dll_out = comp.dll
c_dll_out_path = build/

cc = cl
bash = bash -c
ccflags = /D_USRDLL /D_WINDLL
ccflags2 = /MT /link /DLL
sources = server.c
c_dll_source = computation.c

cudacc = nvcc
cuda_dll_source = cudaarray.cu
cuda_libs = lib/*
cuda_obj = cuda.obj

main: $(sources)
	$(cc) -o $@ $^
	del *.obj

run:
	.\main.exe

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