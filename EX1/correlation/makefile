# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME

# Source files
SRCS = correlation.c ../../utilities/polybench.c
OBJS = correlation.o ../../utilities/polybench.o

# Output executable
TARGET = correlation

# Default rule
all: $(TARGET)

# Compile correlation.c into an object file
correlation.o: correlation.c
	$(CC) $(CFLAGS) -c correlation.c -o correlation.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f correlation.o ../../utilities/polybench.o $(TARGET)
