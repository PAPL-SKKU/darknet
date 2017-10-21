CXX:=nvcc
CFLAGS:=-O3 -D__MAIN_TEST__ -D__DEBUG_ON__ -DOPENDNN
LDFLAGS:=

# OBJ must include a .cpp and .h with same name
OBJ:=
EXE:=NumberADT.exe

all: $(EXE)

# Rule for object files, each object must have a header
$(OBJ): %.o: %.cu %.hpp
	$(CXX) -c $< -o $@ $(CFLAGS)

%.o: %.cu
	$(CXX) -c $< -o $@ $(CFLAGS)

# Rule for executables
$(EXE): %.exe: %.o $(OBJ)
	$(CXX) $^ -o $@ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *.exe
