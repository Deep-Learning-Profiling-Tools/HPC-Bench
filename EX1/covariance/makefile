# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME

# Source files
SRCS = covariance.c ../../utilities/polybench.c
OBJS = covariance.o ../../utilities/polybench.o

# Output executable
TARGET = covariance

# Default rule
all: $(TARGET)

# Compile covariance.c into an object file
covariance.o: covariance.c
	$(CC) $(CFLAGS) -c covariance.c -o covariance.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f covariance.o ../../utilities/polybench.o $(TARGET)
