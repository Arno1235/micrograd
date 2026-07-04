#include <iostream>
#include "micrograd/engine.cpp"

int main() {
    
    std::cout << "\n--- Test 1 ---" << std::endl;

    Value v1 = Value(1);
    Value v2 = Value(2);
    Value v3 = v1.add(v2);
    std::cout << v1.data << std::endl;
    std::cout << v2.data << std::endl;
    std::cout << v3.data << std::endl;

    std::cout << "\n--- Test 2 ---" << std::endl;
    
    
    return 0;
}
