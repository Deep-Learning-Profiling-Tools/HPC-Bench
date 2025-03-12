# Compiler and flags
CC = gcc
CFLAGS = -O3 -I ../../utilities/ -I . 

# Source files
SRCS = 3mm.c ../../utilities/polybench.c
OBJS = 3mm.o ../../utilities/polybench.o

# Output executable
TARGET =3mm

# Default rule
all: $(TARGET)

# Compile 3mm.c into an object file
3mm.o: 3mm.c
	$(CC) $(CFLAGS) -c 3mm.c -o 3mm.o

# Compile polybench.c into an object file
../../utilities/polybench.o: ../../utilities/polybench.c
	$(CC) $(CFLAGS) -c ../../utilities/polybench.c -o ../../utilities/polybench.o

# Link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -o $(TARGET)

# Clean up generated files
clean:
	rm -f 3mm.o ../../utilities/polybench.o $(TARGET)
