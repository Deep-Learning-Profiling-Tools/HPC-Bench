# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME

# Source files
SRCS = doitgen.c ../../utilities/polybench.c
OBJS = doitgen.o ../../utilities/polybench.o

# Output executable
TARGET = doitgen

# Default rule
all: $(TARGET)

# Compile doitgen.c into an object file
doitgen.o: doitgen.c
	$(CC) $(CFLAGS) -c doitgen.c -o doitgen.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f doitgen.o ../../utilities/polybench.o $(TARGET)

