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

	/* 
	 * Nu se proceseaza nimic daca indexul thread-ului e mai mare decat numarul de elemente
	 * care trebuie introduse in hashtable
	 */
	if (idx >= numKeys) {
		return;
	}

	// Obtin indexul din array unde ar trebui adaugat elementul
	int hashcode = getHash(keys[idx], hashTable->capacity);
	int old = 0;

		// Cauta primul slot liber din array
		while (1) {
			// Obtine atomic elementul de pe slotul incercat 
			old = atomicCAS(&hashTable->elements[hashcode].key, EMPTY_SLOT, keys[idx]);
			/* 
			 * Daca slotul e gol, doar se adauga elementul
			 * Daca slotul contine deja cheia, se updateaza valoarea
			 */
			if (old == EMPTY_SLOT || old == keys[idx]) {
				hashTable->elements[hashcode].value = values[idx];
				// Se retine numarul de elemente din hashtable
				atomicAdd(&hashTable->size, 1);
				return;
			}
			// Trece la slotul urmator daca cel curent este ocupat
			hashcode = (hashcode + 1)  % (hashTable->capacity - 1);
		}
}

__global__ void kernel_get(HashTable* hashTable, int* keys, int* values, int numKeys) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	/* 
	 * Nu se proceseaza nimic daca indexul thread-ului e mai mare decat numarul de valori
	 * care trebuie extrase in hashtable
	 */
	if (idx >= numKeys) {
		return;
	}

	// Obtin indexul din array de unde ar trebui extrasa valoarea
	int hashcode = getHash(keys[idx], hashTable->capacity);

	// Cauta cheia de la indexul curent
	while(1) {
		// Verifica sa coincida cheia cautat cu cea de la hash-ul curent
		if (hashTable->elements[hashcode].key == keys[idx]) {
			values[idx] = hashTable->elements[hashcode].value;
			return;
		}
		hashcode = (hashcode + 1)  % (hashTable->capacity - 1);
	}
	

}

__global__ void kernel_reshape(Elem* newElements, int newCapacity, HashTable* oldHashTable) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	/* 
	 * Nu se proceseaza nimic daca: 
	 **	indexul thread-ului e mai mare decat numarul de elemente din hashtable
	 **	slotul curent este gol pentru ca inseamna ca nu a fost populat inca
	 */
	if (idx >= oldHashTable->capacity || oldHashTable->elements[idx].key == EMPTY_SLOT) {
		return;
	}

	// Hashcode-ul se calculeaza in functie de noua capacitate
	int hashcode = getHash(oldHashTable->elements[idx].key, newCapacity);
	int old;
	
	// Se cauta un slot liber
	while(1) {
		old = atomicCAS(&newElements[hashcode].key, EMPTY_SLOT, oldHashTable->elements[idx].key);
		/* 
		 * Se adauga doar daca slotul este gol pentru ca se garanteaza ca nu vor exista
		 * chei duplicate
		 */
		if (old == EMPTY_SLOT) {
			atomicCAS(&newElements[hashcode].value, EMPTY_SLOT, oldHashTable->elements[idx].value);
			return;
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
	
	// Aloca structura de HashTable
	glbGpuAllocator->_cudaMallocManaged((void**) &hashTable, sizeof(HashTable));
	cudaCheckError();

	// Numarul maxim de elemente din hashtable
	hashTable->capacity = size;
	// Numarul de elemente ocupate din hashtable
	hashTable->size = 0;

	// Aloc array-ul de perechi
	glbGpuAllocator->_cudaMalloc((void**) &(hashTable->elements), size * sizeof(Elem));
	cudaCheckError();
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
	int numBlocks;

	// Aloca un nou array de perechi de dimensiunea updatata
	Elem* newElements;
	glbGpuAllocator->_cudaMalloc((void**) &newElements, numBucketsReshape * sizeof(Elem));
	cudaCheckError();
	
	/*
	 * Daca e nevoie de reshape, dar hashtable-ul este gol nu se mai apeleaza functia
	 * de kernel, doar se muta referinta array-ului de perechi catre noul array
	 * si se modifica capacitatea
	 *
	 */
	if (hashTable->size == 0) {
		glbGpuAllocator->_cudaFree(hashTable->elements);
		hashTable->elements = newElements;
		hashTable->capacity = numBucketsReshape;
		return;
	}
	
	// Calculeaza parametri necesari rularii kernelului
 	numBlocks = hashTable->capacity / BLOCK_SIZE;
	// Caz in care block-ul final nu este complet
	if (hashTable->capacity % BLOCK_SIZE) {
		numBlocks++;
	}

	// Muta perechile in noul array
	kernel_reshape<<<numBlocks, BLOCK_SIZE>>> (newElements, numBucketsReshape, hashTable);
	cudaDeviceSynchronize();

	cudaCheckError();
	
	// Elibereaza memoria ocupata de array-ul vechi si updateaza campurile hashtable-ului 
	glbGpuAllocator->_cudaFree(hashTable->elements);
	hashTable->elements = newElements;
	hashTable->capacity = numBucketsReshape; 

}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int numBlocks;
	int *keysDev, *valuesDev;

	// Aloca memorie pentru array-urile de chei si de valori in GPU
	glbGpuAllocator->_cudaMalloc((void **) &keysDev, numKeys * sizeof(int));
	cudaCheckError();

	glbGpuAllocator->_cudaMalloc((void **) &valuesDev, numKeys * sizeof(int));
	cudaCheckError();
	
	// Copiaza valorile si cheile din host pe device
	cudaMemcpy(keysDev, keys , numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valuesDev, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	
	// Calculeaza parametri necesari rularii kernelului
	numBlocks = numKeys / BLOCK_SIZE;
	// Caz in care block-ul final nu este complet
	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}

	// Calculeaza dimensiunea dupa adaugarea cheilor 
	float newSize = (float)(hashTable->size + numKeys); 
	// Verifica daca viitoare dimensiune depaseste limita maxima de ocupare
	if (newSize / hashTable->capacity >= MAX_LOAD_FACTOR) {
		// Calculeaza noua capacitate astfel incat load factorul sa ramana "decent"
		int updatedCapacity = (newSize / DECENT_LOAD_FACTOR) + 1;
		reshape(updatedCapacity);
	}
	
	// Apeleaza kernelul care se ocupa de inserarea efectiva
	kernel_insert_key<<<numBlocks, BLOCK_SIZE>>> (keysDev, valuesDev, numKeys, hashTable);
	cudaDeviceSynchronize();
	
	// Elibereaza memoria ocupata de array-urile alocate pe device
	glbGpuAllocator->_cudaFree(keysDev);
	glbGpuAllocator->_cudaFree(valuesDev);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int numBlocks;
	int *keysDev, *values;

	// Aloca pe device un array in care vor fi mutate cheile
	glbGpuAllocator->_cudaMalloc((void **) &keysDev, numKeys * sizeof(int));
	cudaCheckError();

	/*
	 * Aloca array-ul si pe host si pe device pentru a-l putea folosi si
	 * in functia de kernel si pentru a-l putea returna in host
	 */
	glbGpuAllocator->_cudaMallocManaged((void **) &values, numKeys * sizeof(int));
	cudaCheckError();
	
	// Copiaza cheile in device
	cudaMemcpy(keysDev, keys , numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	// Calculeaza parametri necesari rularii kernelului
	numBlocks = numKeys / BLOCK_SIZE;
	// Caz in care block-ul final nu este complet
	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}

	// Obtine array-ul de valori
	kernel_get<<<numBlocks, BLOCK_SIZE>>> (hashTable, keysDev, values, numKeys);
	cudaDeviceSynchronize();

	cudaCheckError();

	// Elibereaza array-ul de valori de pe device
	glbGpuAllocator->_cudaFree(keysDev);

	return values;
}

/*
* Functie nefolosita, implementata pentru a respecta cerinta
*/
float GpuHashTable::loadFactor() {
	return (float)hashTable->size / hashTable->capacity;
}