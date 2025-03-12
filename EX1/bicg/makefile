# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME

# Source files
SRCS = bicg.c ../../utilities/polybench.c
OBJS = bicg.o ../../utilities/polybench.o

# Output executable
TARGET = bicg

# Default rule
all: $(TARGET)

# Compile bicg.c into an object file
bicg.o: bicg.c
	$(CC) $(CFLAGS) -c bicg.c -o bicg.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f bicg.o ../../utilities/polybench.o $(TARGET)
