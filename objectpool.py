from collections import deque
import weakref

class ObjectPool:
    def __init__(self, create_func, reset_func=None, initial_size=0):
        self.create_func = create_func
        self.reset_func = reset_func
        self.pool = []
        self.active_objects = weakref.WeakSet()

        # Initialize the pool with initial objects
        for _ in range(initial_size):
            self.pool.append(self.create_func())

    def get(self):
        if self.pool:
            obj = self.pool.pop()
        else:
            obj = self.create_func()

        self.reset_func(obj)
        self.active_objects.add(obj)
        return obj

    def release(self, obj):
        if obj in self.active_objects:
            self.active_objects.remove(obj)
            if self.reset_func:
                self.reset_func(obj)
            self.pool.append(obj)

    def clear(self):
        self.pool.clear()
        self.active_objects.clear()