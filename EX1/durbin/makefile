# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME 

# Source files
SRCS = durbin.c ../../utilities/polybench.c
OBJS = durbin.o ../../utilities/polybench.o

# Output executable
TARGET = durbin

# Default rule
all: $(TARGET)

# Compile durbin.c into an object file
durbin.o: durbin.c
	$(CC) $(CFLAGS) -c durbin.c -o durbin.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f durbin.o ../../utilities/polybench.o $(TARGET)
