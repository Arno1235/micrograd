#include <iostream>
#include <functional>
#include <tuple>
#include "micrograd/nn.cpp"


std::tuple<Value*, double> loss(MLP* model, std::vector<std::vector<double>> Xb, std::vector<double> yb) {

    std::vector<std::vector<Value*>> inputs = {};

    for (std::vector<double> xrow : Xb) {
        
        std::vector<Value*> temp = {};
        for (double itm : xrow) temp.insert(temp.end(), new Value(itm));

        inputs.insert(inputs.end(), temp);
    }

    // forward the model to get scores
    std::vector<Value*> scores = {};
    for (std::vector<Value*> input : inputs) scores.insert(scores.end(), model->call(input));

    // svm "max-margin" loss
    std::vector<Value*> losses = {};
    for (int i = 0; i < yb.size(); i++) losses.insert(losses.end(), scores[i]->mul(yb[i])->neg()->add(1.0)->relu());

    Value* data_loss = new Value(0);
    for (Value* l : losses) data_loss = data_loss->add(l);
    data_loss = data_loss->mul(1.0 / losses.size());

    // L2 regularization
    Value* alpha = new Value(0.0001);
    Value* temp = new Value(0);
    for (Value* p : model->parameters()) temp = temp->add(p->mul(p));
    Value* reg_loss = alpha->mul(temp);
    std::cout << "data loss " << data_loss->data << " reg loss " << reg_loss->data << std::endl;
    Value* total_loss = data_loss->add(reg_loss);

    // also get accuracy
    std::vector<int> accuracy = {};
    for (int i = 0; i < yb.size(); i++) {
        accuracy.insert(accuracy.end(), (yb[i] > 0) == (scores[i]->data > 0) ? 1 : 0);
    }

    double acc = 0;
    for (int a : accuracy) acc += a;
    acc /= accuracy.size();

    return  std::make_tuple(total_loss, acc);
}


std::vector<double> get_y() {
    std::vector<double> y = {0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0};
    for (int i = 0; i < y.size(); i++) y[i] = y[i]*2-1;
    return y;
}

std::vector<std::vector<double>> get_X() {
    std::vector<std::vector<double>> X = {};

    X.insert(X.end(), {1.12211461, 0.08147717});
    X.insert(X.end(), {-0.81882941 , 0.05879006});
    X.insert(X.end(), { 1.61370966 ,-0.1246459 });
    X.insert(X.end(), {-0.92300918 , 0.3652289 });
    X.insert(X.end(), {0.14385146, 0.04438005});
    X.insert(X.end(), {0.16447247, 0.11738346});
    X.insert(X.end(), { 1.33877062 ,-0.23800993});
    X.insert(X.end(), { 0.87114861 ,-0.42271759});
    X.insert(X.end(), { 1.83129946 ,-0.14104383});
    X.insert(X.end(), {0.4875712 , 0.63909283});
    X.insert(X.end(), {0.03746235, 0.42358809});
    X.insert(X.end(), {-0.44391685 , 0.89673931});
    X.insert(X.end(), {-0.81222949 , 0.91209092});
    X.insert(X.end(), { 1.63552312 ,-0.34999676});
    X.insert(X.end(), {0.47353904, 0.9573426 });
    X.insert(X.end(), {0.75354932, 0.62372714});
    X.insert(X.end(), { 0.26421282 ,-0.24241983});
    X.insert(X.end(), { 1.42755726 ,-0.37251036});
    X.insert(X.end(), {-0.37235606 , 0.95669171});
    X.insert(X.end(), {-0.96130197 , 0.32609011});
    X.insert(X.end(), {0.78085847, 0.7974894 });
    X.insert(X.end(), { 0.91660903 ,-0.42763844});
    X.insert(X.end(), { 1.04703809 ,-0.54449247});
    X.insert(X.end(), {-0.06036305 , 0.11960909});
    X.insert(X.end(), {0.02918954, 0.306839  });
    X.insert(X.end(), {-0.39573225 , 0.89654389});
    X.insert(X.end(), {-0.10464591 , 1.11788313});
    X.insert(X.end(), {1.88110004, 0.29920257});
    X.insert(X.end(), {0.82740878, 0.34497717});
    X.insert(X.end(), { 1.29777112 ,-0.36654315});
    X.insert(X.end(), {-0.67689285 , 0.85559957});
    X.insert(X.end(), {0.52952995, 0.94735594});
    X.insert(X.end(), {-0.84380229 , 0.60473982});
    X.insert(X.end(), {0.26598471, 0.88732199});
    X.insert(X.end(), {0.13740386, 0.39785689});
    X.insert(X.end(), {-0.91043936 ,-0.09709664});
    X.insert(X.end(), { 1.33740031 ,-0.36741197});
    X.insert(X.end(), { 1.02257719 ,-0.39752649});
    X.insert(X.end(), { 1.02490132 ,-0.5486393 });
    X.insert(X.end(), {-0.7508959  , 0.25328772});
    X.insert(X.end(), {1.20281632, 0.08115382});
    X.insert(X.end(), {-0.46910212 , 0.78079622});
    X.insert(X.end(), {0.74083668, 0.45923254});
    X.insert(X.end(), {0.78690512, 0.76624746});
    X.insert(X.end(), {-0.13005191 , 1.1193894 });
    X.insert(X.end(), { 0.80402306 ,-0.42315447});
    X.insert(X.end(), { 0.28330367 ,-0.21944071});
    X.insert(X.end(), {-0.7110554,  0.7116388});
    X.insert(X.end(), { 0.30262452 ,-0.09338909});
    X.insert(X.end(), {0.80791463, 0.33653833});
    X.insert(X.end(), {-0.94162691 , 0.16801858});
    X.insert(X.end(), { 1.14081485 ,-0.46289718});
    X.insert(X.end(), {-0.15713752 , 0.93210664});
    X.insert(X.end(), { 1.7150437  ,-0.18362009});
    X.insert(X.end(), { 0.37246575 ,-0.12607841});
    X.insert(X.end(), {-0.74309578 , 0.69895167});
    X.insert(X.end(), {0.68142489, 0.68563424});
    X.insert(X.end(), { 0.86861248 ,-0.37278617});
    X.insert(X.end(), {1.00229575, 0.21687478});
    X.insert(X.end(), {-1.02219391 , 0.39750821});
    X.insert(X.end(), {-0.61076258 , 0.83117137});
    X.insert(X.end(), {-0.77668305 , 0.64360932});
    X.insert(X.end(), {1.10530001, 0.21916281});
    X.insert(X.end(), {-0.1789046  , 1.06959774});
    X.insert(X.end(), {0.40498306, 0.82647834});
    X.insert(X.end(), {1.814573  , 0.03452538});
    X.insert(X.end(), {-0.79128563 , 0.20006881});
    X.insert(X.end(), {1.98173149, 0.46076054});
    X.insert(X.end(), { 0.73242796 ,-0.39965754});
    X.insert(X.end(), {2.11141647, 0.18064071});
    X.insert(X.end(), {2.16205598, 0.49423382});
    X.insert(X.end(), {0.89624542, 0.46153606});
    X.insert(X.end(), {0.37349376, 1.04498301});
    X.insert(X.end(), { 0.58496455 ,-0.32125984});
    X.insert(X.end(), {0.17632979, 0.19745572});
    X.insert(X.end(), {0.10911565, 0.48155701});
    X.insert(X.end(), {0.29753275, 0.99583639});
    X.insert(X.end(), { 0.09604657 ,-0.04626986});
    X.insert(X.end(), { 0.47464364 ,-0.10512647});
    X.insert(X.end(), { 1.11634711 ,-0.41553437});
    X.insert(X.end(), { 0.55362759 ,-0.42312582});
    X.insert(X.end(), {0.1895281 , 1.01835655});
    X.insert(X.end(), { 1.94566914 ,-0.0953034 });
    X.insert(X.end(), {-0.08462662 , 1.07262342});
    X.insert(X.end(), { 1.16858008 ,-0.02848107});
    X.insert(X.end(), {0.12446847, 1.05725031});
    X.insert(X.end(), {2.03389924, 0.2847298 });
    X.insert(X.end(), {-0.02800054 , 0.17076764});
    X.insert(X.end(), {0.74093481, 0.41411438});
    X.insert(X.end(), {0.79394703, 0.5597258 });
    X.insert(X.end(), { 0.86874262 ,-0.5301472 });
    X.insert(X.end(), { 1.61874435 ,-0.32584529});
    X.insert(X.end(), { 1.42986564 ,-0.4733421 });
    X.insert(X.end(), { 1.97480435 ,-0.17793162});
    X.insert(X.end(), {1.85356347, 0.34226396});
    X.insert(X.end(), {1.74912164, 0.02833902});
    X.insert(X.end(), {-0.68566889 , 0.46535694});
    X.insert(X.end(), {1.75237435, 0.16452051});
    X.insert(X.end(), { 0.18078955 ,-0.00029542});
    X.insert(X.end(), {0.12108297, 1.06555225});

    return X;
}


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

    Value* total_loss;
    double acc;
    std::tie(total_loss, acc) = loss(model, get_X(), get_y());
    std::cout << "loss " << total_loss->data << ", accuracy " << acc*100 << "%" << std::endl;

    for (int k = 0; k < 100; k++) {

        // forward
        std::tie(total_loss, acc) = loss(model, get_X(), get_y());

        // backward
        model->zero_grad();
        total_loss->backward();

        // update (sgd)
        double learning_rate = 1.0 - 0.9*k/100;
        for (Value* p : model->parameters()) p->data -= learning_rate * p->grad;

        std::cout << "step " << k << " loss " << total_loss->data << ", accuracy " << acc*100 << "% lr " << learning_rate << std::endl;
    }
    
    return 0;
}
