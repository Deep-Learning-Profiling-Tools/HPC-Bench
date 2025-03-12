# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME 

# Source files
SRCS = trisolv.c ../../utilities/polybench.c
OBJS = trisolv.o ../../utilities/polybench.o

# Output executable
TARGET = trisolv

# Default rule
all: $(TARGET)

# Compile doitgen.c into an object file
trisolv.o: trisolv.c
	$(CC) $(CFLAGS) -c trisolv.c -o trisolv.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f trisolv.o ../../utilities/polybench.o $(TARGET)
