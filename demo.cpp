#include <iostream>
#include <functional>
// #include "micrograd/engine.cpp"
#include "micrograd/nn.cpp"

int main() {
    
    std::cout << "\n--- Test 1 ---" << std::endl;

    Value* a = new Value(-4);
    Value* b = new Value(2);
    Value* c = a->add(b);
    Value* d = a->mul(b)->add(b->pow(3));
    c = c->add(c->add(1));
    c = c->add(c->add(1)->sub(a));
    d = d->add(d->mul(2)->add(b->add(a)->relu()));
    d = d->add(d->mul(3)->add(b->sub(a)->relu()));
    Value* e = c->sub(d);
    Value* f = e->pow(2);
    Value* g = f->div(2);
    g = g->add(f->rdiv(10));

    std::cout << g->data << std::endl;  // prints 24.7041, the outcome of this forward pass

    g->backward();

    std::cout << a->grad << std::endl;  // prints 138.8338, i.e. the numerical value of dg/da
    std::cout << b->grad << std::endl;  // prints 645.5773, i.e. the numerical value of dg/db

    std::cout << "\n--- Test 2 ---" << std::endl;

    // initialize a model 
    MLP* model = new MLP(2, {16, 16, 1});  // 2-layer neural network
    std::cout << "number of parameters " << model->parameters().size() << std::endl;  // expect 337 parameters

    
    return 0;
}
