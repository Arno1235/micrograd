#include <iostream>
#include <functional>
#include <cmath>
#include <set>

class Value {
    public:
        double data;
        double grad;
        double _pow_value;

        std::function<void(Value* value)> _backward;
        std::vector<Value*> _prev;

        std::vector<Value*> _topo;
        std::set<Value*> _visited;

        Value(double test);
        Value(double test, std::vector<Value*> _children);
        void constructor(double test, std::vector<Value*> _children);

        Value add(double other);
        Value add(Value *other);
        Value mul(double other);
        Value mul(Value *other);
        Value pow(double other);
        Value relu();
        void _build_topo(Value* value);
        void backward();
};

Value::Value(double test) {
    this->constructor(test, {});
}

Value::Value(double test, std::vector<Value*> _children) {
    this->constructor(test, _children);
}

void Value::constructor(double test, std::vector<Value*> _children) {
    this->grad = 0;
    this->_pow_value = 0;
    this->data = test;
    this->_prev = _children;
    this->_backward = []( Value* value ){ };
}

Value Value::add(Value* other) {

    Value out = Value(this->data + other->data, {this, other});
    out._backward = []( Value* value ){
        value->_prev[0]->grad += value->grad;
        value->_prev[1]->grad += value->grad;
    };

    return out;
}

Value Value::add(double other) {
    Value temp = Value(other);
    return add(&temp);
}

Value Value::mul(Value* other) {

    Value out = Value(this->data * other->data, {this, other});
    out._backward = []( Value* value ){
        value->_prev[0]->grad += value->_prev[1]->data * value->grad;
        value->_prev[1]->grad += value->_prev[0]->data * value->grad;
    };

    return out;
}

Value Value::mul(double other) {
    Value temp = Value(other);
    return mul(&temp);
}

Value Value::pow(double other) {

    Value out = Value(std::pow(this->data, other), {this});
    out._pow_value = other;
    out._backward = []( Value* value ){
        value->_prev[0]->grad += (value->_pow_value * std::pow(value->_prev[0]->data, value->_pow_value - 1)) * value->grad;
    };

    return out;
}

Value Value::relu() {

    double new_data = this->data;
    if (this->data < 0) new_data = 0;

    Value out= Value(new_data, {this});
    out._backward = []( Value* value ){
        if (value->data > 0) value->_prev[0]->grad += value->grad;
    };

    return out;
}

void Value::_build_topo(Value* value) {
    if (this->_visited.count(value) == 0) {
        this->_visited.insert(value);
        for (Value* child : value->_prev) {
            this->_build_topo(child);
        }
        this->_topo.insert(this->_topo.begin(), value);
    }
}

void Value::backward() {

    this->_topo.clear();
    this->_visited.clear();

    this->_build_topo(this);

    this->grad = 1;
    for (Value* v : this->_topo) {
        v->_backward(v);
    }

}
