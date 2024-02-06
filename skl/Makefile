KERNEL_PATH   := ./kernel
OUTPUT_PATH   := ./out

CC     := icc
CFLAGS := -O3 -qopenmp -restrict
CFLAGS += -wd3950 -Wall -Werror
CFLAGS += -xCORE-AVX512

$(OUTPUT_PATH)/dgemm_test_mkl.out: $(KERNEL_PATH)/dgemm_test.c $(KERNEL_PATH)/mkl.c
	$(CC) -o $@ $^ $(CFLAGS) -DKERNEL=\"MKL\" -mkl

$(OUTPUT_PATH)/dgemm_test_openblas.out: $(KERNEL_PATH)/dgemm_test.c $(KERNEL_PATH)/openblas.c $(OPENBLASROOT)/libopenblas_skylakex.a
	$(CC) -o $@ $^ $(CFLAGS) -DKERNEL=\"OPENBLAS\" -DUSE_OPENBLAS_HEADER

$(OUTPUT_PATH)/dgemm_test_blis.out: $(KERNEL_PATH)/dgemm_test.c $(KERNEL_PATH)/blis_skl.c $(BLISROOT)/lib/skx/libblis.a
	$(CC) -o $@ $^ $(CFLAGS) -DKERNEL=\"BLIS\"

$(OUTPUT_PATH)/dgemm_test_userdgemm.out: $(KERNEL_PATH)/dgemm_test.c $(KERNEL_PATH)/userdgemm.c $(KERNEL_PATH)/skl_ksnt.c $(KERNEL_PATH)/skl_nstn.c
	$(CC) -o $@ $^ $(CFLAGS) -DKERNEL=\"USERDGEMM\" -mkl -DVERIFY

mkl: $(OUTPUT_PATH)/dgemm_test_mkl.out

openblas: $(OUTPUT_PATH)/dgemm_test_openblas.out

blis: $(OUTPUT_PATH)/dgemm_test_blis.out

userdgemm: $(OUTPUT_PATH)/dgemm_test_userdgemm.out

all: mkl openblas blis userdgemm

clean:
	rm -f $(OUTPUT_PATH)/dgemm_test_*.out
