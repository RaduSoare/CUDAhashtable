#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define EMPTY_SLOT 0
#define BLOCK_SIZE 256
#define MAX_LOAD_FACTOR 0.8f
#define MIN_LOAD_FACTOR 0.5f
#define DECENT_LOAD_FACTOR 0.7f

typedef struct {
	int key;
	int value;
} Elem;

typedef struct {
	int size;
	int capacity;
	Elem* elements;
} HashTable;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	HashTable* hashTable;
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		float getLoadFactor();
	
		~GpuHashTable();
};

#endif
