#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

#define FOLDER_PATH "./data" // 모니터링할 폴더 경로

void process_bin_file(const char *file_path) {
    // 파일 처리 로직
    printf("Processing file: %s\n", file_path);

    // 바이너리 파일 읽기
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        perror("fopen");
        return;
    }

    // 파일 크기 확인
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // 파일 데이터 읽기
    file_size = 8 * sizeof(float);
    unsigned char *buffer = malloc(file_size);
    
    if (buffer == NULL) {
        perror("malloc");
        fclose(file);
        return;
    }
    fread(buffer, 1, file_size, file);

    // 데이터 처리 (예: 데이터 출력)
    printf("File size: %ld bytes\n", file_size);
    for (int i = 0; i < file_size; i++) {
        printf("%02X ", buffer[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // 메모리 해제 및 파일 닫기
    free(buffer);
    fclose(file);

    // 파일 삭제
    if (remove(file_path) == 0) {
        printf("Deleted file: %s\n", file_path);
    } else {
        perror("remove");
    }
}

int main() {
    while (1) {
        struct dirent *entry;
        DIR *dir = opendir(FOLDER_PATH);

        if (dir == NULL) {
            perror("opendir");
            return EXIT_FAILURE;
        }

        int found_file = 0; // 파일이 있는지 확인하기 위한 플래그
        while ((entry = readdir(dir)) != NULL) {
            // 숨김 파일(`.` 및 `..`) 무시
            if (entry->d_name[0] == '.') {
                continue;
            }

            // .bin 파일만 처리
            const char *ext = strrchr(entry->d_name, '.');
            if (ext == NULL || strcmp(ext, ".bin") != 0) {
                continue; // 확장자가 .bin이 아니면 건너뜀
            }

            // 파일 경로 생성
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", FOLDER_PATH, entry->d_name);
            
            // 파일 처리
            process_bin_file(file_path);

            found_file = 1; // 파일이 있었음을 표시
            break;          // 첫 번째 파일만 처리 후 종료
        }

        closedir(dir);

        if (!found_file) {
            printf("No files found. Waiting...\n");
            sleep(1);
        }

        sleep(1); // 1초 대기
    }

    return EXIT_SUCCESS;
}
