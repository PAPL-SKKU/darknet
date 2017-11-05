CXX:=nvcc
CFLAGS:=-O3 -D__DEBUG_ON__ -DOPENDNN -D__TEST__
LDFLAGS:=

# OBJ must include a .cpp and .h with same name
OBJ:=NumberADT.o
EXE:=test.exe

all: $(EXE)

# Rule for object files, each object must have a header
$(OBJ): %.o: %.cpp %.hpp
	$(CXX) -c $< -o $@ $(CFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

# Rule for executables
$(EXE): %.exe: %.o $(OBJ)
	$(CXX) $^ -o $@ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *.exe
