# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . 

# Source files
SRCS = nussinov.c ../../utilities/polybench.c
OBJS = nussinov.o ../../utilities/polybench.o

# Output executable
TARGET = nussinov

# Default rule
all: $(TARGET)

# Compile nussinov.c into an object file
nussinov.o: nussinov.c
	$(CC) $(CFLAGS) -c nussinov.c -o nussinov.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f nussinov.o ../../utilities/polybench.o $(TARGET)
