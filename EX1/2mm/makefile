# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . 

# Source files
SRCS = 2mm.c ../../utilities/polybench.c
OBJS = 2mm.o ../../utilities/polybench.o

# Output executable
TARGET = 2mm

# Default rule
all: $(TARGET)

# Compile 2mm.c into an object file
2mm.o: 2mm.c
	$(CC) $(CFLAGS) -c 2mm.c -o 2mm.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f 2mm.o ../../utilities/polybench.o $(TARGET)
