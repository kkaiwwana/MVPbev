import numpy as np
cimport numpy as np
cimport cython
cimport cqueue

cdef class Queue:

    cdef cqueue.Queue* _c_queue

    def __cinit__(self):
        self._c_queue = cqueue.queue_new()
        if self._c_queue is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_queue is not NULL:
            cqueue.queue_free(self._c_queue)

    cpdef int append(self, int value) except -1:
        if not cqueue.queue_push_tail(self._c_queue, <void*>value):
            raise MemoryError()
        return 0

    cdef int extend(self, int* values, Py_ssize_t count) except -1:
        cdef Py_ssize_t i
        for i in range(count):
            if not cqueue.queue_push_tail(self._c_queue, <void*>values[i]):
                raise MemoryError()
        return 0

    cpdef int peek(self) except? 0:
        cdef int value = <int>cqueue.queue_peek_head(self._c_queue)
        if value == 0:
            # this may mean that the queue is empty, or that it
            # happens to contain a 0 value
            if cqueue.queue_is_empty(self._c_queue):
                raise IndexError("Queue is empty")
        return value

    cpdef int pop(self) except? 0:
        cdef int value = <int>cqueue.queue_pop_head(self._c_queue)
        if value == 0:
            # this may mean that the queue is empty, or that it
            # happens to contain a 0 value
            if cqueue.queue_is_empty(self._c_queue):
                raise IndexError("Queue is empty")
        return value

    def __bool__(self):    # same as __nonzero__ in Python 2.x
        return not cqueue.queue_is_empty(self._c_queue)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.int_t, ndim=2] _bfs(
    np.ndarray[np.int_t, ndim=2] semantic_map,
    int x, 
    int y, 
    int key, 
    np.ndarray[np.int_t, ndim=2] visited,
    int max_len
):
    cdef np.ndarray[np.int_t, ndim=2] region
    cdef int region_len = 0
    cdef np.ndarray[np.int_t, ndim=2] seen
    
    cdef Queue x_q = Queue()
    cdef Queue y_q = Queue()
    x_q.append(x + 1) 
    y_q.append(y + 1)
    
    seen = visited.copy()
    region = np.zeros((max_len, 2), dtype=np.int64)
    
    while x_q:
        x, y = x_q.pop(), y_q.pop()
        x -= 1
        y -= 1
        if semantic_map[x, y] == key:
            region[region_len] = np.array([x, y])
            region_len += 1
            visited[x, y] = 1
        else:
            continue
        
        # aviod zero indices
        if x < semantic_map.shape[0] - 1:
            if visited[x + 1, y] != 1 and seen[x + 1, y] != 1:
                x_q.append(x + 1 + 1)
                y_q.append(y + 1)
                seen[x + 1, y] = 1
        if y < semantic_map.shape[1] - 1:
            if visited[x, y + 1] != 1 and seen[x, y + 1] != 1:
                x_q.append(x + 1)
                y_q.append(y + 1 + 1)
                seen[x, y + 1] = 1
        if x != 0:
            if visited[x - 1, y] != 1 and seen[x - 1, y] != 1:
                x_q.append(x - 1 + 1)
                y_q.append(y + 1)
                seen[x - 1, y] = 1
        if y != 0:
            if visited[x, y - 1] != 1 and seen[x, y - 1] != 1:
                x_q.append(x + 1)
                y_q.append(y - 1 + 1)
                seen[x, y - 1] = 1

    return region[:region_len]
    

def non_recur_bfs(seman_map, x, y, key, visited, max_len=50000):
    return _bfs(seman_map, x, y, key, visited, max_len)