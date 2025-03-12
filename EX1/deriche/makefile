# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . 

# Source files
SRCS = deriche.c ../../utilities/polybench.c
OBJS = deriche.o ../../utilities/polybench.o

# Output executable
TARGET = deriche

# Default rule
all: $(TARGET)

# Compile deriche.c into an object file
deriche.o: deriche.c
	$(CC) $(CFLAGS) -c deriche.c -o deriche.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f deriche.o ../../utilities/polybench.o $(TARGET)
