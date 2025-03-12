CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . -DPOLYBENCH_TIME

# Source files
SRCS = adi.c ../../utilities/polybench.c
OBJS = adi.o ../../utilities/polybench.o

# Output executable
TARGET = adi

# Default rule
all: $(TARGET)

# Compile 2mm.c into an object file
adi.o: adi.c
	$(CC) $(CFLAGS) -c adi.c -o adi.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f adi.o ../../utilities/polybench.o $(TARGET)
