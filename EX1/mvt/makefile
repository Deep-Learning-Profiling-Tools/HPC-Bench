# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME

# Source files
SRCS = mvt.c ../../utilities/polybench.c
OBJS = mvt.o ../../utilities/polybench.o

# Output executable
TARGET = mvt

# Default rule
all: $(TARGET)

# Compile doitgen.c into an object file
mvt.o: mvt.c
	$(CC) $(CFLAGS) -c mvt.c -o mvt.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f mvt.o ../../utilities/polybench.o $(TARGET)
