#include <iostream>


class Value {
    public:
        double data;
        double grad;

        Value(double test);

        Value add(double other);
        Value add(Value other);
};

Value::Value(double test) {
    this->data = test;
}

Value Value::add(Value other) {
    Value out = Value(this->data + other.data);

    return out;
}

Value Value::add(double other) {
    return add(Value(other));
}
