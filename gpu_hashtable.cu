#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/

// Thomas Mueller in https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__ int getHash(int key, int capacity) {
	key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;

	return key % capacity;
}

__global__ void kernel_insert_key(int *keys, int* values, int numKeys, HashTable* hashTable) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numKeys) {
		return;
	}

	
	// Obtin elementul din array unde trebuie adaugat elementul
	int hashcode = getHash(keys[idx], hashTable->capacity);
	bool foundEmptySlot = false;
	int old = 0;
	
	// Cauta primul slot liber din array
	while (!foundEmptySlot) {
		// Obtine atomic elementul de pe slotul incercat 
		int old = atomicCAS(&hashTable->elements[hashcode].key, EMPTY_SLOT, keys[idx]);
		if (old == EMPTY_SLOT || old == keys[idx]) {
			atomicCAS(&hashTable->elements[hashcode].value, EMPTY_SLOT, values[idx]);
			foundEmptySlot = true;
		}
		// Trece la slotul urmator daca cel curent este ocupat
		hashcode = (hashcode + 1)  % (hashTable->capacity - 1);
	}

	// Mareste size-ul doar daca elementul a fost adaugat pe un slot gol
	if (old == EMPTY_SLOT) {
		atomicAdd(&hashTable->size, 1);
	}
	
	//printf("%d %d %d %d\n", idx, hashcode, hashTable->elements[hashcode].key, hashTable->elements[hashcode].value);
		
}

__global__ void kernel_get(HashTable* hashTable, int* keys, int* values, int numKeys) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numKeys) {
		return;
	}

	int hashcode = getHash(keys[idx], hashTable->capacity);

	bool foundKey = false;
	while(!foundKey) {
		if (hashTable->elements[hashcode].key == keys[idx]) {
			values[idx] = hashTable->elements[hashcode].value;
			foundKey = true;
		}
		hashcode = (hashcode + 1)  % (hashTable->capacity - 1);
	}
	

}

__global__ void kernel_reshape(Elem* newElements, int newCapacity, HashTable* oldHashTable) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= oldHashTable->capacity) {
		return;
	}
	

	int hashcode = getHash(oldHashTable->elements[idx].key, newCapacity);
	//printf("%d %d %d %d\n",idx, hashcode, oldHashTable->elements[idx].key, oldHashTable->elements[idx].value);
	bool rehashedKey = false;

	while(!rehashedKey) {
		int old = atomicCAS(&newElements[hashcode].key, EMPTY_SLOT, oldHashTable->elements[idx].key);
		if (old == EMPTY_SLOT) {
			atomicCAS(&newElements[hashcode].value, EMPTY_SLOT, oldHashTable->elements[idx].value);
			rehashedKey = true;
		}
		hashcode = (hashcode + 1)  % (newCapacity - 1);
	}

}




/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t rc;
	
	rc = glbGpuAllocator->_cudaMallocManaged((void**) &hashTable, sizeof(HashTable));
	if (rc != cudaSuccess) {
		cout << "HashMap Malloc Failed!" << endl;
		return;
	}

	// Numarul maxim de elemente din hashtable
	hashTable->capacity = size;
	// // Numarul de elemente ocupate din hashtable
	hashTable->size = 0;

	//Aloc array-ul de bucket-uri (array de liste)
	rc = glbGpuAllocator->_cudaMalloc((void**) &(hashTable->elements), size * sizeof(Elem));
	if (rc != cudaSuccess) {
		cout << "Elements Malloc Failed!" << endl;
		return;
	}


}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(hashTable->elements);
	glbGpuAllocator->_cudaFree(hashTable);
}


/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t rc;
	int numBlocks;

	Elem* newElements;
	rc = glbGpuAllocator->_cudaMalloc((void**) &newElements, numBucketsReshape * sizeof(Elem));
	if (rc != cudaSuccess) {
		cout << "Elements Malloc Failed!" << endl;
		return;
	}
	

	if (hashTable->size == 0) {
		glbGpuAllocator->_cudaFree(hashTable->elements);
		hashTable->elements = newElements;
		hashTable->capacity = numBucketsReshape;
		cout << "era gol" << endl;

		return;
	}
	
 	numBlocks = hashTable->capacity / BLOCK_SIZE;
	
	// Caz in care block-ul final nu este complet
	if (hashTable->capacity % BLOCK_SIZE) {
		numBlocks++;
	}
	
	
	kernel_reshape<<<numBlocks, BLOCK_SIZE>>> (newElements, numBucketsReshape, hashTable);
	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
	

	glbGpuAllocator->_cudaFree(hashTable->elements);
	 //glbGpuAllocator->_cudaFree(hashTable);
	hashTable->elements = newElements;
	hashTable->capacity = numBucketsReshape; 

}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t rc;
	int numBlocks;
	// Aloca memorie pentru array-urile de chei si de valori in GPU
	int *keysDev, *valuesDev;


	rc = glbGpuAllocator->_cudaMalloc((void **) &keysDev, numKeys * sizeof(int));
	if (rc != cudaSuccess) {
		perror("keysDev Malloc Failed!");
		return false;
	}
	rc = glbGpuAllocator->_cudaMalloc((void **) &valuesDev, numKeys * sizeof(int));
	if (rc != cudaSuccess) {
		perror("valuesDev Malloc Failed!");
		return false;
	}
	
	cudaMemcpy(keysDev, keys , numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valuesDev, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	numBlocks = numKeys / BLOCK_SIZE;

	// Caz in care block-ul final nu este complet
	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}

	// Caz cand e nevoie de rehash
	if ((float)(hashTable->size + numKeys) / hashTable->capacity >= MAX_LOAD_FACTOR) {
		// Calculeaza noua capacitate
		int updatedCapacity = ((float)(hashTable->size + numKeys) / MAX_LOAD_FACTOR) + 1;
		cout << "Trebuie resize "<< updatedCapacity << endl;
		fprintf(stdout, "%d\n", updatedCapacity);
		reshape(updatedCapacity);
		
		
	}
	
	//cout << hashTable->size << " " << hashTable->capacity << endl;
	kernel_insert_key<<<numBlocks, BLOCK_SIZE>>> (keysDev, valuesDev, numKeys, hashTable);
	cudaDeviceSynchronize();
	//cout << hashTable->size << " " << hashTable->capacity << endl;
	

	glbGpuAllocator->_cudaFree(keysDev);
	glbGpuAllocator->_cudaFree(valuesDev);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t rc;
	int numBlocks;
	int *keysDev, *values;


	rc = glbGpuAllocator->_cudaMalloc((void **) &keysDev, numKeys * sizeof(int));
	if (rc != cudaSuccess) {
		perror("keysDev Malloc Failed!");
		return NULL;
	}
	rc = glbGpuAllocator->_cudaMallocManaged((void **) &values, numKeys * sizeof(int));
	if (rc != cudaSuccess) {
		perror("valuesDev Malloc Failed!");
		return NULL;
	}
	
	// for (int i = 0; i < numKeys; i++) {
	// 	cout << keys[i] << endl;
	// }

	cudaMemcpy(keysDev, keys , numKeys * sizeof(int), cudaMemcpyHostToDevice);

	numBlocks = numKeys / BLOCK_SIZE;

	// Caz in care block-ul final nu este complet
	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}
	kernel_get<<<numBlocks, BLOCK_SIZE>>> (hashTable, keysDev, values, numKeys);
	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}

	glbGpuAllocator->_cudaFree(keysDev);

	return values;
}
