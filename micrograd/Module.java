package micrograd;

import java.util.List;
import java.util.ArrayList;

public class Module {

    public void zero_grad() {
        for (Value p : this.parameters()) p.grad = 0.0d;
    }

    public List<Value> parameters() {
        return new ArrayList<>();
    }
    
}
