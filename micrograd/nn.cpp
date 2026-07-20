#include "engine.cpp"
#include <random>

class Module {
    public:
        void zero_grad();
        std::vector<Value*> parameters();
};

void Module::zero_grad() {
    for (Value* p : this->parameters()) {
        p->grad = 0;
    }
}

std::vector<Value*> Module::parameters() {
    return {};
}

class Neuron : Module {
    public:
        std::vector<Value*> w;
        Value* b;
        bool nonlin;

        Neuron(int nin, bool nonlin);

        std::vector<Value*> parameters();
        Value* call(std::vector<Value*> x);
};

Neuron::Neuron(int nin, bool nonlin=true) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    this->w.clear();
    for (int i = 0; i<nin; i++) {
        this->w.insert(this->w.end(), new Value(dist(e2)));
    }

    this->b = new Value(0);
    this->nonlin = nonlin;
}

Value* Neuron::call(std::vector<Value*> x) {

    Value* act = this->w[0]->mul(x[0]);

    for (int i = 1; i < this->w.size(); i++) {
        act = act->add(this->w[i]->mul(x[i]));
    }

    if (this->nonlin) {
        return act->relu();
    }
    return act;
}

std::vector<Value*> Neuron::parameters() {
    std::vector<Value*> p = this->w;
    p.insert(p.end(), this->b);
    return p;
}

class Layer : Module {
    public:
        std::vector<Neuron*> neurons;

        Layer(int nin, int nout, bool nonlin);

        std::vector<Value*> parameters();
        std::vector<Value*> call(std::vector<Value*> x);
};

Layer::Layer(int nin, int nout, bool nonlin=true) {
    this->neurons.clear();
    for (int i = 0; i<nout; i++) {
        this->neurons.insert(this->neurons.end(), new Neuron(nin, nonlin));
    }
}

std::vector<Value*> Layer::call(std::vector<Value*> x) {
    std::vector<Value*> out = {};
    for (Neuron* n : this->neurons) {
        out.insert(out.end(), n->call(x));
    }
    return out;
}

std::vector<Value*> Layer::parameters() {
    std::vector<Value*> p = {};
    for (Neuron* n : this->neurons) {
        std::vector<Value*> np = n->parameters();
        p.insert(p.end(), np.begin(), np.end());
    }
    return p;
}

class MLP : Module {
    public:
        std::vector<Layer*> layers;

        MLP(int nin, std::vector<int> nouts);

        void zero_grad();
        std::vector<Value*> parameters();
        Value* call(std::vector<Value*> x);
};

MLP::MLP(int nin, std::vector<int> nouts) {
    std::vector<int> sz = {nin};
    sz.insert(sz.end(), nouts.begin(), nouts.end());

    this->layers.clear();
    for (int i = 0; i<nouts.size(); i++) {
        bool nonlin = i != nouts.size() - 1;
        this->layers.insert(this->layers.end(), new Layer(sz[i], sz[i+1], nonlin));
    }
}

Value* MLP::call(std::vector<Value*> x) {
    for (Layer* layer : this->layers) {
        x = layer->call(x);
    }
    return x[0];
}

std::vector<Value*> MLP::parameters() {
    std::vector<Value*> p = {};
    for (Layer* l : this->layers) {
        std::vector<Value*> lp = l->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    }
    return p;
}

void MLP::zero_grad() {
    for (Value* p : this->parameters()) {
        p->grad = 0;
    }
}

