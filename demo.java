// to run:
// javac demo.java && java demo

import micrograd.Module;
import micrograd.MLP;
import micrograd.Layer;
import micrograd.Neuron;
import micrograd.Value;

import java.util.ArrayList;

public class demo {
    public static void main(String args[]) {
        System.out.println("Hello World!");

        Value v = new Value(2.0d);
        Value out = v.mul(3.0d);

        System.out.println(Double.toString(out.data));
        System.out.println(Double.toString(v.grad));

        out.grad = 1.0d;
        out._backward.accept(out);

        System.out.println(Double.toString(v.grad));
        
    }
}
