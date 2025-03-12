# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME 

# Source files
SRCS = atax.c ../../utilities/polybench.c
OBJS = atax.o ../../utilities/polybench.o

# Output executable
TARGET = atax

# Default rule
all: $(TARGET)

# Compile atax.c into an object file
atax.o: atax.c
	$(CC) $(CFLAGS) -c atax.c -o atax.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f atax.o ../../utilities/polybench.o $(TARGET)
