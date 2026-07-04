
# micrograd

https://github.com/karpathy/micrograd

### Rules

No AI! Only Stackoverflow:
```
<search-term> site:stackoverflow.com
```

run:
```
demo.ipynb
```

output:
```
# Test 1
24.7041
138.8338
645.5773

# Test 2
MLP of [Layer of [ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2)], Layer of [ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16)], Layer of [LinearNeuron(16)]]
number of parameters 337

Value(data=1.4155524118353182, grad=0) 0.48

step 0 loss 1.4155524118353182, accuracy 48.0%
step 1 loss 0.48470950016045783, accuracy 83.0%
step 2 loss 0.40904549154949726, accuracy 79.0%
step 3 loss 0.6283767081194753, accuracy 82.0%
step 4 loss 0.4057385617781459, accuracy 85.0%
step 5 loss 0.33292063832107727, accuracy 86.0%
step 6 loss 0.2944384773907324, accuracy 89.0%
step 7 loss 0.27330830160258585, accuracy 89.0%
step 8 loss 0.2677397878082909, accuracy 90.0%
step 9 loss 0.26874157195213505, accuracy 89.0%
step 10 loss 0.26387729237584323, accuracy 90.0%
step 11 loss 0.29558034704008174, accuracy 88.0%
step 12 loss 0.3583038754745555, accuracy 87.0%
step 13 loss 0.2964476292570516, accuracy 88.0%
step 14 loss 0.30726830052724974, accuracy 90.0%
step 15 loss 0.24291180611519952, accuracy 89.0%
step 16 loss 0.2332999702022026, accuracy 91.0%
step 17 loss 0.2264966241484159, accuracy 90.0%
step 18 loss 0.22179964019584983, accuracy 92.0%
step 19 loss 0.2241022286083921, accuracy 90.0%
step 20 loss 0.2189324500760828, accuracy 93.0%
step 21 loss 0.2126253042009747, accuracy 90.0%
step 22 loss 0.2085182727556198, accuracy 93.0%
step 23 loss 0.20850177578764736, accuracy 90.0%
step 24 loss 0.2377606446568348, accuracy 90.0%
step 25 loss 0.22586841624030407, accuracy 88.0%
step 26 loss 0.24345782143971761, accuracy 90.0%
step 27 loss 0.20872645457284045, accuracy 90.0%
step 28 loss 0.20460383318234915, accuracy 93.0%
step 29 loss 0.18256400680021628, accuracy 90.0%
step 30 loss 0.18004080100597014, accuracy 93.0%
step 31 loss 0.17739683526352884, accuracy 92.0%
step 32 loss 0.18312217565903818, accuracy 93.0%
step 33 loss 0.16937271969068488, accuracy 92.0%
step 34 loss 0.17614250505823967, accuracy 93.0%
step 35 loss 0.15689465536107527, accuracy 93.0%
step 36 loss 0.1705627081921114, accuracy 93.0%
step 37 loss 0.13203881797554723, accuracy 95.0%
step 38 loss 0.12714763927726053, accuracy 95.0%
step 39 loss 0.13469774770594778, accuracy 94.0%
step 40 loss 0.15301813000543396, accuracy 94.0%
step 41 loss 0.1246093158650168, accuracy 95.0%
step 42 loss 0.13218961019006575, accuracy 95.0%
step 43 loss 0.11352526022154943, accuracy 97.0%
step 44 loss 0.12157736308775367, accuracy 95.0%
step 45 loss 0.1080489351761052, accuracy 97.0%
step 46 loss 0.10739244224058571, accuracy 95.0%
step 47 loss 0.10112988376543644, accuracy 97.0%
step 48 loss 0.09942582796993371, accuracy 95.0%
step 49 loss 0.09621213639694423, accuracy 97.0%
step 50 loss 0.09110814510492277, accuracy 95.0%
step 51 loss 0.07999831868847801, accuracy 97.0%
step 52 loss 0.08186892789636668, accuracy 96.0%
step 53 loss 0.08677460682447101, accuracy 98.0%
step 54 loss 0.08634557951408997, accuracy 96.0%
step 55 loss 0.0714466511767467, accuracy 97.0%
step 56 loss 0.06408826037556187, accuracy 97.0%
step 57 loss 0.06630445979487963, accuracy 99.0%
step 58 loss 0.08012302874272063, accuracy 97.0%
step 59 loss 0.07543657432673605, accuracy 99.0%
step 60 loss 0.08292799398867795, accuracy 96.0%
step 61 loss 0.059506806390855996, accuracy 97.0%
step 62 loss 0.05788691415378297, accuracy 99.0%
step 63 loss 0.068859619120714, accuracy 97.0%
step 64 loss 0.04996943300284852, accuracy 99.0%
step 65 loss 0.047425312693626695, accuracy 98.0%
step 66 loss 0.04863840106045009, accuracy 100.0%
step 67 loss 0.06462850600760589, accuracy 97.0%
step 68 loss 0.044704665084381666, accuracy 100.0%
step 69 loss 0.04294979292002347, accuracy 98.0%
step 70 loss 0.03730214402260622, accuracy 100.0%
step 71 loss 0.03679696464068417, accuracy 99.0%
step 72 loss 0.044505222703532785, accuracy 100.0%
step 73 loss 0.054948720455488986, accuracy 97.0%
step 74 loss 0.03505050834862204, accuracy 100.0%
step 75 loss 0.03243844575388958, accuracy 100.0%
step 76 loss 0.03321701763636739, accuracy 99.0%
step 77 loss 0.03179615617114036, accuracy 100.0%
step 78 loss 0.03324940892630026, accuracy 99.0%
step 79 loss 0.03130747519100872, accuracy 100.0%
step 80 loss 0.03125789452328496, accuracy 100.0%
step 81 loss 0.030709268910353985, accuracy 100.0%
step 82 loss 0.029049833397347784, accuracy 100.0%
step 83 loss 0.02876604091239881, accuracy 100.0%
step 84 loss 0.028582936037718994, accuracy 100.0%
step 85 loss 0.02714873939350471, accuracy 100.0%
step 86 loss 0.02658146769694441, accuracy 100.0%
step 87 loss 0.0270731760701509, accuracy 100.0%
step 88 loss 0.027715436993566524, accuracy 100.0%
step 89 loss 0.025135498848704606, accuracy 100.0%
step 90 loss 0.02513186195220514, accuracy 100.0%
step 91 loss 0.023986927753584304, accuracy 100.0%
step 92 loss 0.02320472795664831, accuracy 100.0%
step 93 loss 0.023507382593164247, accuracy 100.0%
step 94 loss 0.022392959275745188, accuracy 100.0%
step 95 loss 0.02237237827961737, accuracy 100.0%
step 96 loss 0.021364207421708976, accuracy 100.0%
step 97 loss 0.020882144094243084, accuracy 100.0%
step 98 loss 0.020476990494657968, accuracy 100.0%
step 99 loss 0.020160713020160464, accuracy 100.0%
100 epochs took 143.62321090698242s -> 1.4362321090698242s/epoch
```

### License

MIT
