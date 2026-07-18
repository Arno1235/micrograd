#include <iostream>
#include <functional>
#include "micrograd/engine.cpp"

int main() {
    
    std::cout << "\n--- Test 1 ---" << std::endl;

    Value a = Value(-4);
    Value b = Value(2);
    Value c = a.add(&b);
    Value d = a.mul(&b).add(b.pow(3));

    c = c.add(c.add(1));
    c = c.add(c.add(1).sub(&a));

    d = d.add(d.mul(2).add(b.add(&a).relu()));
    d = d.add(d.mul(3).add(b.sub(&a).relu()));

    Value e = c.sub(&d);
    Value f = e.pow(2);
    Value g = f.div(2);
    g = g.add(f.rdiv(10));

    std::cout << g.data << std::endl;

    g.backward();

    std::cout << a.grad << std::endl;
    std::cout << b.grad << std::endl;

    std::cout << "\n--- Test 2 ---" << std::endl;
    
    return 0;
}
