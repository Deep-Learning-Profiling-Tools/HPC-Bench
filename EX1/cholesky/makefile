# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . 

# Source files
SRCS = cholesky.c ../../utilities/polybench.c
OBJS = cholesky.o ../../utilities/polybench.o

# Output executable
TARGET = cholesky

# Default rule
all: $(TARGET)

# Compile cholesky.c into an object file
cholesky.o: cholesky.c
	$(CC) $(CFLAGS) -c cholesky.c -o cholesky.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f cholesky.o ../../utilities/polybench.o $(TARGET)
