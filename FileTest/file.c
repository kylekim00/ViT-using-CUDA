#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *file = fopen("data.bin", "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return EXIT_FAILURE;
    }

    // Assuming we know the dimensions and type of the data
    int rows = 2;
    int cols = 3;
    float *array = malloc(rows * cols * sizeof(float));

    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    // Read the data
    size_t read = fread(array, sizeof(float), rows * cols, file);
    if (read != rows * cols) {
        fprintf(stderr, "Failed to read enough elements (expected %d, got %zu)\n", rows * cols, read);
        free(array);
        fclose(file);
        return EXIT_FAILURE;
    }

    // Print the data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", array[i * cols + j]);
        }
        printf("\n");
    }

    // Clean up
    free(array);
    fclose(file);

    return EXIT_SUCCESS;
}
