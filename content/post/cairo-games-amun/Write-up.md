```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
```

# Amun
## Solving the last Cairo-Games challenge

For their second edition of the [Cairo Games](https://www.cairo-lang.org/the-cairo-games/puzzles2/), StarkWare have proposed 5 challenges for testing your Cairo skills.

Cairo is a specialized language used for ....
You write a program for verifying a solution to a problem, but you embed some hints in the source code which the compiler uses to set all memory locations properly.
Within the context of the Cairo Games, the goal is to write the appropriate hints, and essentially solve the problem that the program is verifying.

## Exploratory Challenge Analysis

// write some background on the type of challenges etc
### Setting things up

The Cairo program starts with the `main()` function, and we'll focus for now on the first half.


```python
func main{output_ptr : felt*, pedersen_ptr : HashBuiltin*, range_check_ptr}():
    alloc_locals
    let (local data : felt*) = get_data()
    let (local sol : felt*) = alloc()
    local sol_size

    let (state : DictAccess*) = alloc()
    local state_start : DictAccess* = state

    run{state=state}(data, sol, sol_size)

    # ...
end

```



The main() function seems to be doing a couple of things:

1. Populate an array `data` with `get_data()`
2. Expect a solution array `sol` of size `sol_size`. 
3. Process the solution using `data`, and storing the result in `state`. Here, `state` is represented as a [DictAccess](https://www.cairo-lang.org/docs/hello_cairo/dict.html?) list. We can think of it as a list of all updates to a dictionary, but we'll come back to this later.
4. _Squash_ `state` into a dictionary of size 48. Since no more updates will be performed on `state`, `squash_dict()` creates a new dictionary containing only the initial and final values for each key that was tracked in `state`.
5. Get some more initial data by calling `get_initial_value` (note the passing of `hash_ptr=pedersen_ptr` as argument).
6. Perform some verification on the `squashed_dict` and `initial_value`

If we try to run the program as-is, we get the following error:
```
Error: code:258:33: While expanding the reference 'sol_size' in:
    run{state=state}(data, sol, sol_size)
                                ^******^
code:253:11: Error at pc=0:849:
Unknown value for memory cell at address 1:7.
    local sol_size
          ^******^
```

At this stage, `sol_size` is not initialized (neither are `sol` and `state` in fact), so this _hints_ at the fact we may have to write our own hints inside this `main` function.



```python
func get_data() -> (res : felt*):
    alloc_locals
    local a0 = 0
    local a = 2
    local a = 4
    # ...
    local a = 28
    local a = 36
    local a = 12
    let (__fp__, _) = get_fp_and_pc()
    return (res=&a0)
end

```

At first sight, this function seems to be overwriting the `a` variable many times. 
In Cairo, memory is immutable, so we aren't actually able to change the value of `a`. 
Instead, we are actually writing a new value in the next memory slot, so that we end up with one contiguous array of data.
The function then returns the adress of `a0` which points to the start of the array. 

To better understand how this works in more detail, I recommend reading [How Cairo Works](https://www.cairo-lang.org/docs/how_cairo_works/index.html).

We'll need to study this data later, so we store it neatly and run some basic analysis on it.


```python
DATA = [
    0, 2, 4, 6, 1, 3, 5, 7, 8, 32, 24, 16, 9, 33, 25, 17, 10, 34, 26, 18, 24, 26, 28, 30, 25, 27, 29, 31, 4, 32, 44, 
    20, 3, 39, 43, 19, 2, 38, 42, 18, 16, 18, 20, 22, 17, 19, 21, 23, 6, 24, 42, 12, 5, 31, 41, 11, 4, 30, 40, 10,
    8, 10, 12, 14, 9, 11, 13, 15, 0, 16, 40, 36, 7, 23, 47, 35, 6, 22, 46, 34, 32, 34, 36, 38, 33, 35, 37, 39, 2, 8,
    46, 28, 1, 15, 45, 27, 0, 14, 44, 26, 40, 42, 44, 46, 41, 43, 45, 47, 22, 30, 38, 14, 21, 29, 37, 13, 20, 28, 36, 12
]
print(f"Item range: [{min(DATA)}, ...,{max(DATA)}]")
print(f"Item count: {len(DATA)}")
counts = [DATA.count(i) for i in range(max(DATA))]
print(f"Occurences of each i = 0, ..., {max(DATA)}: {counts}")
```

    Item range: [0, ...,47]
    Item count: 120
    Occurences of each i = 0, ..., 47: [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]


### Running the solution


```python
func run{range_check_ptr, state : DictAccess*}(data : felt*, sol : felt*, sol_size : felt):
    if sol_size == 0:
        return ()
    end
    tempvar x = [sol]
    assert_nn_le(x, 5)
    cycle(data + 20 * x)
    cycle(data + 20 * x + 4)
    cycle(data + 20 * x + 8)
    cycle(data + 20 * x + 12)
    cycle(data + 20 * x + 16)

    run(data=data, sol=sol + 1, sol_size=sol_size - 1)
    return ()
end
```

`run` is a recursive function that iterates over all elements in the array `sol`.
At each iteration, it sets `x` to the next element, and exits when it has visited the whole list.

The first thing to notice is `assert_nn_le(x, 5)`, which gives us our first hint about the solution we need to construct:

**Each element in `sol` must be in the range \[0, 5\]**

Next, each loop of `run()` calls `cycle()` 5 times with varying inputs.
Looking ahead, we see that `cycle()` only reads 4 elements of the array it is given, which means we are feeding it 5 blocks of 4 values, or 20 values in total.
Therefore, each element of `sol` references a specific row of `data` which will be applied to the `state`. 

We verify that each row in `DATA` contains unique items:


```python
BLOCKS = [DATA[i:i+20] for i in range(0, len(DATA), 20)]
for i in range(len(BLOCKS)):
    row = BLOCKS[i]
    assert len(set(row)) == len(row), f"row {i} contains duplicates"
print("no duplicates in rows")
```

    no duplicates in rows


As we mentioned earlier, the `state` variable is meant to represent an 'append-only dictionary'.
Concretely, `state` is a list of `DictAccess` items which are appended each time an update to the dictionary occurs.
The `DictAccess` struct is defined in [starkware/cairo/common/dict_access.cairo](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/cairo/common/dict_access.cairo), but can be 



```python
from dataclasses import dataclass

@dataclass
class DictAccess:
    key: int
    prev_value: int
    new_value: int
```

First, we create 4 local references to the next `DictAccess` elements, which represent the dictionary updates we are about to perform.
The keys that will be updated are defined by the 4 values in `values`, so we set the `key`s of each `DictAccess` accordingly.
Finally we apply a cyclic permutation to the 4 keys by setting each `new_value` to be the `prev_value` of another key.


```python
func cycle{state : DictAccess*}(values : felt*):
    # Create local references to the 4 dictionary updates
    let state0 = state
    let state1 = state0 + DictAccess.SIZE
    let state2 = state1 + DictAccess.SIZE
    let state3 = state2 + DictAccess.SIZE
    # We are operating on the keys defined in values
    assert state0.key = [values + 0]
    assert state1.key = [values + 1]
    assert state2.key = [values + 2]
    assert state3.key = [values + 3]
    # Apply a cyclic permutation to the keys
    assert state0.new_value = state3.prev_value
    assert state1.new_value = state0.prev_value
    assert state2.new_value = state1.prev_value
    assert state3.new_value = state2.prev_value
    let state = state + 4 * DictAccess.SIZE
    return ()
end
```

The nature of Cairo makes it harder to reason about what a specific block of code actually does. 
Here's some equivalent Python code that produces the same intended result. 
The main difference with the original Cairo code is that we are working directly with dictionaries.


```python
# getKeys returns the block of 4 keys defined by DATA
def getKeys(row, block: int) -> list[int]:
    assert 0 <= row and row <= 5, "row must be in [0,5]"
    assert 0 <= block and block <= 4, "block must be in [0,4]"
    block_idx = 20*row + 4*block
    return DATA[block_idx:block_idx+4]

# run iterates over all elements in the solution, and 
def run(state: dict[int], data: list[int], sol: list[int]):
    for s in sol:
        assert 0 <= s and s <=5, "sol contains an element outside [0,5]"
        cycle(state, getKeys(s, 0))
        cycle(state, getKeys(s, 1))
        cycle(state, getKeys(s, 2))
        cycle(state, getKeys(s, 3))
        cycle(state, getKeys(s, 4))

# cycle permutes the values of 4 elements in state[keys] in the following way:
# 
# 0 -> 3
# 1 -> 0
# 2 -> 1
# 3 -> 2
def cycle(state: dict[int], keys: list[int]):
    key0, key1, key2, key3 = keys
    state[key0], state[key1], state[key2], state[key3] = state[key3], state[key0], state[key1], state[key2]
```

Taking a step back, we start to wonder what this `run()` function actually does.
Let's recap the information we learned up til now:

1. `state` is a dictionary that contains some initial data referenced by the keys 0, ..., 48 .
2. `data` can be seen as a 6x20 matrix where each row contains a subset of {0, ..., 48} .
3. when iterating over `sol`, each element references row of `data` which itself references the keys that will be permuted.
4. since all rows of `data` are duplicate-free, we can view them as permutations over {0, ..., 48} .
5. each permutation is a composition of 5 cyclic permutations of order 4 each, so all permutation have order 4 (by _order_ we mean that applying the permutation 4 times is the same as doing nothing).

Before getting too much ahead of ourselve, we're going to check out the verification part of the problem, looking at the `get_initial_value()` and `verify()` functions.

### Verification 

After generating the appropriate `state` from the solution, we are interested in the second half of `main()` which is responsible for verifying our provided solution was correct. 
Hopefully, understanding what this code does will give us more hints about how we should construct our solution.


```python
func main{output_ptr : felt*, pedersen_ptr : HashBuiltin*, range_check_ptr}():
    # ...

    # 1.
    let (local squashed_dict : DictAccess*) = alloc()
    let (squashed_dict_end) = squash_dict(
        dict_accesses=state_start, dict_accesses_end=state, squashed_dict=squashed_dict)
    assert squashed_dict_end = squashed_dict + DictAccess.SIZE * 48
    local range_check_ptr = range_check_ptr

    # 2.
    let (initial_value) = get_initial_value{hash_ptr=pedersen_ptr}()
    local pedersen_ptr : HashBuiltin* = pedersen_ptr
    # 3.
    verify(squashed_dict=squashed_dict, initial_value=initial_value, n=0)

    let output_ptr = output_ptr + 1
    return ()
end
```

We separate the code snippet in 3 parts and briefly explain what each one does.

1.
The `state` that we have produced isn't particularly useful in its current form, since it represents all updates we performed along the way. 
Applying `squash_dict()` to `state` create a list of tuples `(key, initial_value, final_value)` sorted by `key`.

The `assert` statement also tells us that the the size of the `squashed_dict` is 48, which means that our solution will affect all keys.

2. `get_initial_value()` most likely provides an array of elements. More on this later.

3. `verify()` should check wether our processed solution is coherent with `initial_value`.

So far, we didn't learn a lot more, so let's dig deeper into the functions.

#### get_initial_value()


```python
func get_initial_value{hash_ptr : HashBuiltin*}() -> (initial_value : felt*):
    alloc_locals
    let (local initial_value : felt*) = alloc()
    assert [initial_value] = 48
    let (res) = hash_chain(initial_value)
    assert res = 402684044838294963951952172461450293510735826065192598384325922359699836469

    let (res) = hash_chain(initial_value + 1)
    assert res = 1508108551069464286813785297355641266663485599320848393798932455588476865295

    let (res) = hash_chain(initial_value + 7)
    assert res = 2245701625176425331085101334837624242646502129018701371434984384296915870715

    let (res) = hash_chain(initial_value + 12)
    assert res = 3560520899812162122215526869789497390123010766571927682749531967294685134040

    let (res) = hash_chain(initial_value + 18)
    assert res = 196997208112053944281778155212956924860955084720008751336605214240056455402

    let (res) = hash_chain(initial_value + 24)
    assert res = 1035226353110224801512289478587695122129015832153304072590365512606504328818

    let (res) = hash_chain(initial_value + 30)
    assert res = 1501038259288321437961590173137394957125779122158200548115283728521438213428

    let (res) = hash_chain(initial_value + 34)
    assert res = 3537881782324467737440957567711773328493014027685577879465936840743865613662

    let (res) = hash_chain(initial_value + 39)
    assert res = 1039623306816876893268944011668782810398555904667703809415056949499773381189

    let (res) = hash_chain(initial_value + 42)
    assert res = 2508528289207660435870821551803296739495662639464901004905339054353214007301

    return (initial_value=initial_value + 1)
end
```

What a mouth full! Let's decompose this bit by bit.

First, `alloc_locals` indicates that we will most likely have to provide hints to the compiler about the actual values in the `initial_value` array.

The first assertion: `assert [initial_value] = 48` nicely tells us that `initial_value[0] == 48`, which is going to help later.
Looking at the return statement however, we see that this value is not actually returned as part of `initial_value`. 

We'll now try to understand the statements of the form:
```
    let (res) = hash_chain(initial_value + idx)
    assert res = ????????
```

The `hash_chain(data_ptr : felt*)` function is defined in [starkware.cairo.common.hash_chain](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/cairo/common/hash_chain.cairo).
Its documentation states:
> Computes a hash chain of a sequence whose length is given at [data_ptr] and the data starts at
> data_ptr + 1. The hash is calculated backwards (from the highest memory address to the lowest).
>
> For example, for the 3-element sequence [x, y, z] the hash is:
>
>   h(3, h(x, h(y, z)))
>
> If data_length = 0, the function does not return (takes more than field prime steps).

The actual hash function used is the "Starkware Pedersen" one defined in [crypto/starkware/crypto/signature/fast_pedersen_hash.py](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/crypto/starkware/crypto/signature/fast_pedersen_hash.py) (it was mentioned in the hints near the end of the [puzzle page](https://www.cairo-lang.org/the-cairo-games/puzzles2/)).

Since everything is slower in Cairo-land, we can use `compute_hash_chain(data)` function from [starkware/cairo/common/hash_chain.py](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/cairo/common/hash_chain.py) to compute hashes in Python directly. 

The two functions are not entirely the same, since the Cairo version assumes that the first element of the array is also its length, whereas the Python one simply computes

`h(data[0], h(data[1], h(..., h(data[n-2], data[n-1]))))`.



Recalling that `initial_value[0] == 48`, we can "word for saying we're making an educated guess" that `len(initial_value) == 49`.

Moreover, the first hash is the result of computing:

`FULL_HASH = h(48, h(initial_value[1], h(initial_value[2], h(..., h(initial_value[48], initial_value[49]))...)))`

We might also guess that the values in `initial_value` are all less than 48, but bruteforcing this would still be intractable.

### verify()


```python
func verify(squashed_dict : felt*, initial_value : felt*, n):
    if n == 6:
        return ()
    end

    assert [squashed_dict + 0] = n * 8
    assert [squashed_dict + 1] = [initial_value]
    assert [squashed_dict + 2] = n

    # ...

    assert [squashed_dict + 21] = n * 8 + 7
    assert [squashed_dict + 22] = [initial_value + 7]
    assert [squashed_dict + 23] = n
    verify(squashed_dict=squashed_dict + 24, initial_value=initial_value + 8, n=n + 1)
    return ()
end
```

The `verify()` function is another recursive function which iterates on both `squashed_dict` and `initial_value`.
We also see that `n = 0, 1, ..., 5`. 

As we might have guessed by now, the `initial_value` array provided by `get_initial_value()` is actually an array representing the starting `state`. 

Therefore, we can confirm that the size of `initial_value` must be 48 as well. 

Looking at the following Python code should give a better understanding of what this function is actually verifying.


```python
FINAL_STATE_VALUES = np.arange(48)//8

def verify(squashed_dict: list[DictAccess], initial_value: list[int]):
    assert len(squashed_dict) == len(initial_value) == 48, "squashed_dict and initial_value must contain 48 elements"
    for key in range(48):
        n = key//8
        state = squashed_dict[key]
        assert n == FINAL_STATE_VALUES[key], "FINAL_STATE_VALUES is not correct"
        assert state.key == key, f"squashed_dict is not sorted at key {i}"
        assert state.prev_value == initial_value[key], f"prev_value for key {key} does not correspond with initial_value"
        assert state.new_value == FINAL_STATE_VALUES[key], f"prev_value for key {key} does not correspond with initial_value"



# verify that our verify function is working correctly
# we create squashed_dict which is already the final state
fake_squashed_dict = [DictAccess(i, v, v) for i, v in enumerate(FINAL_STATE_VALUES)]
verify(fake_squashed_dict, FINAL_STATE_VALUES)

pd.DataFrame(FINAL_STATE_VALUES).T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



The above snippet contained the last clue we are going to extract from the source code. 

Up until now, we knew the keys in our `state` were the integers 0, 1, ..., 47, but we had little information about what values they represented.

By looking at what our final dictionary should look like, we realize that the values are simply integers between 0 and 5 (which all appear exactly 8 times). 

This really restricts the possible outputs of `get_initial_value()`, which means we might actually be able to bruteforce something!

## Breaking `get_initial_value()`


We can now start seriously attacking this problem head-on!
Our goal in this section is to intelligently bruteforce the array returned by `get_initial_value()`.
Here's what we have figured out:

1. At the start of the function `initial_value` is an array of length 49, and `initial_value[0] == 48`.
2. The array returned is `initial_value[1:]`, and it contains only numbers in `[0,5]`.
3. More specifically, each number in `[0,5]` appears exactly 8 times in `initial_value[1:]`

Let's start by formatting the problem's data into Python.


```python
from starkware.cairo.common.hash_chain import compute_hash_chain, pedersen_hash
import itertools
```


```python
FULL_HASH = 402684044838294963951952172461450293510735826065192598384325922359699836469
GIVEN_HASHES = {
    1: 1508108551069464286813785297355641266663485599320848393798932455588476865295,
    7: 2245701625176425331085101334837624242646502129018701371434984384296915870715,
    12: 3560520899812162122215526869789497390123010766571927682749531967294685134040,
    18: 196997208112053944281778155212956924860955084720008751336605214240056455402,
    24: 1035226353110224801512289478587695122129015832153304072590365512606504328818,
    30: 1501038259288321437961590173137394957125779122158200548115283728521438213428,
    34: 3537881782324467737440957567711773328493014027685577879465936840743865613662,
    39: 1039623306816876893268944011668782810398555904667703809415056949499773381189,
    42: 2508528289207660435870821551803296739495662639464901004905339054353214007301
}
```

Recalling the definitition of `hash_chain` (Cairo) and `compute_hash_chain` (Python), 
we can link them together like this:

`hash_chain(data)` = `compute_hash_chain([data[0], data[1], ..., data[data[0]])`

The `data` array must contain only 0, 1, ..., 5, and `data[0]` defines how many elements to hash.
In other words, `data` must be one of the following, where `i_(j,k)` is in `[0,5]` 

```
[ 0 ]
[ 1, i_(1,1) ]
[ 2, i_(2,1), i_(2,2) ]
[ 3, i_(3,1), i_(3,2), i_(3,3) ]
[ 4, i_(4,1), i_(4,2), i_(4,3), i_(4,4) ]
[ 5, i_(5,1), i_(5,2), i_(5,3), i_(5,4), i_(5,5) ]
```

A simple calculation reveals that this yields exactly 9331 possiblities, a very reasonable amount of iteration we are willing to sacrifice.


```python
%%time

working_values = [48] + 48*[-1]

hashes = [h  for idx, h in GIVEN_HASHES.items()]

hash_preimages = {}

# create all tuples (j, i_1, i_2, ..., i_j) with i, j in [0,5]
def tuple_iterator():
    for j in range(6):
        for t in itertools.product(range(6), repeat=j):
            yield (j, *t)

for preimage in tuple_iterator():
    h = compute_hash_chain(preimage)
    if h in hashes:
        print(f"found preimage for {h}: {preimage}")
        hash_preimages[h] = preimage
        if len(hash_preimages) == len(hashes):
            print("\nFound all preimages for the hashes in GIVEN_HASHES!")
            break
```

    found preimage for 1501038259288321437961590173137394957125779122158200548115283728521438213428: (2, 2, 5)
    found preimage for 1039623306816876893268944011668782810398555904667703809415056949499773381189: (2, 3, 4)
    found preimage for 1035226353110224801512289478587695122129015832153304072590365512606504328818: (3, 2, 2, 1)
    found preimage for 196997208112053944281778155212956924860955084720008751336605214240056455402: (4, 0, 1, 5, 5)
    found preimage for 2245701625176425331085101334837624242646502129018701371434984384296915870715: (4, 0, 4, 1, 0)
    found preimage for 3560520899812162122215526869789497390123010766571927682749531967294685134040: (4, 0, 5, 4, 2)
    found preimage for 3537881782324467737440957567711773328493014027685577879465936840743865613662: (4, 1, 1, 3, 2)
    found preimage for 2508528289207660435870821551803296739495662639464901004905339054353214007301: (4, 3, 0, 5, 5)
    found preimage for 1508108551069464286813785297355641266663485599320848393798932455588476865295: (5, 0, 0, 3, 3, 1)
    
    Found all preimages for the hashes in GIVEN_HASHES!
    CPU times: user 8.72 s, sys: 36.8 ms, total: 8.76 s
    Wall time: 8.79 s



```python
# update working_values with the preimages we found
for idx, h in GIVEN_HASHES.items():
    preimage = hash_preimages[h]
    for i, val in enumerate(preimage):
        working_values[idx+i] = val
print(f"Unfortunately, we are still missing {working_values.count(-1)} values...")
```

    Unfortunately, we are still missing 7 values...


We were able to find pre-images for all the hashes, but unfortunately we are still missing 7 values in the `initial_value` array.

Remembering that each number in `[0,5]` appears exactly 8 times in `initial_value`, we can actually find what they are!


```python
# find indices of values we haven't found yet
missing_idx = [i for i, v in enumerate(working_values) if v == -1]

# find the values we are missing, knowing that each number in [0,5] appears 8 times in the solution
missing_values = []
for v in range(6):
    # number of occurences of v=0,...,5
    c = working_values.count(v)
    # add v to missing_values as many times as necessary
    for _ in range(8-c):
        missing_values.append(v)
print(f"We are missing values: {missing_values} at indices {missing_idx}")
```

    We are missing values: [0, 1, 1, 2, 3, 3, 5] at indices [17, 23, 28, 29, 33, 47, 48]


The last step will consist in trying all 7! (5040) permutation of `[0, 1, 1, 2, 3, 3, 5]` as possible missing values for `initial_value`.

While this is half the number of iterations required for breaking all hashes in `GIVEN_HASHES`,
each call to `compute_hash_chain(initial_value)` now performs 48 different hashes.

With our fingers crossed, we run the final bruteforce.


```python
%%time

for possibility in itertools.permutations(missing_values):
    for i, idx in enumerate(missing_idx):
        working_values[idx] = possibility[i]
    h = compute_hash_chain(working_values)
    if h == FULL_HASH:
        break
print("FINISHED\ninitial_value = ", working_values[1:])
```

    FINISHED
    initial_value =  [5, 0, 0, 3, 3, 1, 4, 0, 4, 1, 0, 4, 0, 5, 4, 2, 3, 4, 0, 1, 5, 5, 1, 3, 2, 2, 1, 0, 1, 2, 2, 5, 2, 4, 1, 1, 3, 2, 2, 3, 4, 4, 3, 0, 5, 5, 5, 3]
    CPU times: user 3min 57s, sys: 1.1 s, total: 3min 58s
    Wall time: 3min 59s


Success!

We are able to find the whole `initial_value` array in less than 5 minutes (along with a full day to come up with the solution).

Let's verify that everything is correct:


```python
INITIAL_VALUES = [5, 0, 0, 3, 3, 1, 4, 0, 4, 1, 0, 4, 0, 5, 4, 2, 3, 4, 0, 1, 5, 5, 1, 3, 2, 2, 1, 0, 1, 2, 2, 5, 2, 4, 1, 1, 3, 2, 2, 3, 4, 4, 3, 0, 5, 5, 5, 3]

assert compute_hash_chain(working_values) == FULL_HASH, "first hash is incorrect"
for idx, h in GIVEN_HASHES.items():
    num_vals = working_values[idx]
    hash_block = working_values[idx:idx+1+num_vals]
    assert compute_hash_chain(hash_block) == h, f"hash at index {idx} is incorrect"
```

## Analysing `run()`

Now that we have the initial values of the `state`, we need to find the actual solution array which solves the problem.

1. The solution constists of a sequence of numbers in 0, 1, ..., 5.
2. Each of these elements references one of the 6 permutations that can be applied to the current state.
3. Each permuations has order 4, and affect exactly 20 of the 48 elements.
4. The 


```python
ROWS = 6
COLS = 48
BLOCK_SIZE = 8
BLOCK_COUNT = 6

PERMUTATIONS = np.empty((ROWS,COLS), dtype=np.int8)

for perm_idx in range(ROWS):
    # initialize each row to the identity permutation [0, 1, ..., 47]
    PERMUTATIONS[perm_idx] = np.arange(COLS)
    row = PERMUTATIONS[perm_idx]
    for i in range(5):
        perm = getKeys(perm_idx, i)
        row[perm[0]], row[perm[1]], row[perm[2]], row[perm[3]] = row[perm[3]], row[perm[0]], row[perm[1]], row[perm[2]]

PERMUTATIONS_DF = pd.DataFrame(PERMUTATIONS)
```

The first look at what elements are actually affected by a permutation.

In the following matrix, each row defines a permutation of {0, 1, ..., 47},
and we color the elements in red if the element is moved, and green if it is left intact.


```python
def color_identity(row):
    return [f"background-color: {'red' if row[i] != i else 'green'}" for i in range(len(row))]

PERMUTATIONS_DF.style.set_caption('Fixed points of the permutations. 🟩 = fixed, 🟥 = moved').apply(color_identity,axis=1)
```


<div>

<style  type="text/css" >
#T_0852c_row0_col0,#T_0852c_row0_col1,#T_0852c_row0_col2,#T_0852c_row0_col3,#T_0852c_row0_col4,#T_0852c_row0_col5,#T_0852c_row0_col6,#T_0852c_row0_col7,#T_0852c_row0_col8,#T_0852c_row0_col9,#T_0852c_row0_col10,#T_0852c_row0_col16,#T_0852c_row0_col17,#T_0852c_row0_col18,#T_0852c_row0_col24,#T_0852c_row0_col25,#T_0852c_row0_col26,#T_0852c_row0_col32,#T_0852c_row0_col33,#T_0852c_row0_col34,#T_0852c_row1_col2,#T_0852c_row1_col3,#T_0852c_row1_col4,#T_0852c_row1_col18,#T_0852c_row1_col19,#T_0852c_row1_col20,#T_0852c_row1_col24,#T_0852c_row1_col25,#T_0852c_row1_col26,#T_0852c_row1_col27,#T_0852c_row1_col28,#T_0852c_row1_col29,#T_0852c_row1_col30,#T_0852c_row1_col31,#T_0852c_row1_col32,#T_0852c_row1_col38,#T_0852c_row1_col39,#T_0852c_row1_col42,#T_0852c_row1_col43,#T_0852c_row1_col44,#T_0852c_row2_col4,#T_0852c_row2_col5,#T_0852c_row2_col6,#T_0852c_row2_col10,#T_0852c_row2_col11,#T_0852c_row2_col12,#T_0852c_row2_col16,#T_0852c_row2_col17,#T_0852c_row2_col18,#T_0852c_row2_col19,#T_0852c_row2_col20,#T_0852c_row2_col21,#T_0852c_row2_col22,#T_0852c_row2_col23,#T_0852c_row2_col24,#T_0852c_row2_col30,#T_0852c_row2_col31,#T_0852c_row2_col40,#T_0852c_row2_col41,#T_0852c_row2_col42,#T_0852c_row3_col0,#T_0852c_row3_col6,#T_0852c_row3_col7,#T_0852c_row3_col8,#T_0852c_row3_col9,#T_0852c_row3_col10,#T_0852c_row3_col11,#T_0852c_row3_col12,#T_0852c_row3_col13,#T_0852c_row3_col14,#T_0852c_row3_col15,#T_0852c_row3_col16,#T_0852c_row3_col22,#T_0852c_row3_col23,#T_0852c_row3_col34,#T_0852c_row3_col35,#T_0852c_row3_col36,#T_0852c_row3_col40,#T_0852c_row3_col46,#T_0852c_row3_col47,#T_0852c_row4_col0,#T_0852c_row4_col1,#T_0852c_row4_col2,#T_0852c_row4_col8,#T_0852c_row4_col14,#T_0852c_row4_col15,#T_0852c_row4_col26,#T_0852c_row4_col27,#T_0852c_row4_col28,#T_0852c_row4_col32,#T_0852c_row4_col33,#T_0852c_row4_col34,#T_0852c_row4_col35,#T_0852c_row4_col36,#T_0852c_row4_col37,#T_0852c_row4_col38,#T_0852c_row4_col39,#T_0852c_row4_col44,#T_0852c_row4_col45,#T_0852c_row4_col46,#T_0852c_row5_col12,#T_0852c_row5_col13,#T_0852c_row5_col14,#T_0852c_row5_col20,#T_0852c_row5_col21,#T_0852c_row5_col22,#T_0852c_row5_col28,#T_0852c_row5_col29,#T_0852c_row5_col30,#T_0852c_row5_col36,#T_0852c_row5_col37,#T_0852c_row5_col38,#T_0852c_row5_col40,#T_0852c_row5_col41,#T_0852c_row5_col42,#T_0852c_row5_col43,#T_0852c_row5_col44,#T_0852c_row5_col45,#T_0852c_row5_col46,#T_0852c_row5_col47{
            background-color:  red;
        }#T_0852c_row0_col11,#T_0852c_row0_col12,#T_0852c_row0_col13,#T_0852c_row0_col14,#T_0852c_row0_col15,#T_0852c_row0_col19,#T_0852c_row0_col20,#T_0852c_row0_col21,#T_0852c_row0_col22,#T_0852c_row0_col23,#T_0852c_row0_col27,#T_0852c_row0_col28,#T_0852c_row0_col29,#T_0852c_row0_col30,#T_0852c_row0_col31,#T_0852c_row0_col35,#T_0852c_row0_col36,#T_0852c_row0_col37,#T_0852c_row0_col38,#T_0852c_row0_col39,#T_0852c_row0_col40,#T_0852c_row0_col41,#T_0852c_row0_col42,#T_0852c_row0_col43,#T_0852c_row0_col44,#T_0852c_row0_col45,#T_0852c_row0_col46,#T_0852c_row0_col47,#T_0852c_row1_col0,#T_0852c_row1_col1,#T_0852c_row1_col5,#T_0852c_row1_col6,#T_0852c_row1_col7,#T_0852c_row1_col8,#T_0852c_row1_col9,#T_0852c_row1_col10,#T_0852c_row1_col11,#T_0852c_row1_col12,#T_0852c_row1_col13,#T_0852c_row1_col14,#T_0852c_row1_col15,#T_0852c_row1_col16,#T_0852c_row1_col17,#T_0852c_row1_col21,#T_0852c_row1_col22,#T_0852c_row1_col23,#T_0852c_row1_col33,#T_0852c_row1_col34,#T_0852c_row1_col35,#T_0852c_row1_col36,#T_0852c_row1_col37,#T_0852c_row1_col40,#T_0852c_row1_col41,#T_0852c_row1_col45,#T_0852c_row1_col46,#T_0852c_row1_col47,#T_0852c_row2_col0,#T_0852c_row2_col1,#T_0852c_row2_col2,#T_0852c_row2_col3,#T_0852c_row2_col7,#T_0852c_row2_col8,#T_0852c_row2_col9,#T_0852c_row2_col13,#T_0852c_row2_col14,#T_0852c_row2_col15,#T_0852c_row2_col25,#T_0852c_row2_col26,#T_0852c_row2_col27,#T_0852c_row2_col28,#T_0852c_row2_col29,#T_0852c_row2_col32,#T_0852c_row2_col33,#T_0852c_row2_col34,#T_0852c_row2_col35,#T_0852c_row2_col36,#T_0852c_row2_col37,#T_0852c_row2_col38,#T_0852c_row2_col39,#T_0852c_row2_col43,#T_0852c_row2_col44,#T_0852c_row2_col45,#T_0852c_row2_col46,#T_0852c_row2_col47,#T_0852c_row3_col1,#T_0852c_row3_col2,#T_0852c_row3_col3,#T_0852c_row3_col4,#T_0852c_row3_col5,#T_0852c_row3_col17,#T_0852c_row3_col18,#T_0852c_row3_col19,#T_0852c_row3_col20,#T_0852c_row3_col21,#T_0852c_row3_col24,#T_0852c_row3_col25,#T_0852c_row3_col26,#T_0852c_row3_col27,#T_0852c_row3_col28,#T_0852c_row3_col29,#T_0852c_row3_col30,#T_0852c_row3_col31,#T_0852c_row3_col32,#T_0852c_row3_col33,#T_0852c_row3_col37,#T_0852c_row3_col38,#T_0852c_row3_col39,#T_0852c_row3_col41,#T_0852c_row3_col42,#T_0852c_row3_col43,#T_0852c_row3_col44,#T_0852c_row3_col45,#T_0852c_row4_col3,#T_0852c_row4_col4,#T_0852c_row4_col5,#T_0852c_row4_col6,#T_0852c_row4_col7,#T_0852c_row4_col9,#T_0852c_row4_col10,#T_0852c_row4_col11,#T_0852c_row4_col12,#T_0852c_row4_col13,#T_0852c_row4_col16,#T_0852c_row4_col17,#T_0852c_row4_col18,#T_0852c_row4_col19,#T_0852c_row4_col20,#T_0852c_row4_col21,#T_0852c_row4_col22,#T_0852c_row4_col23,#T_0852c_row4_col24,#T_0852c_row4_col25,#T_0852c_row4_col29,#T_0852c_row4_col30,#T_0852c_row4_col31,#T_0852c_row4_col40,#T_0852c_row4_col41,#T_0852c_row4_col42,#T_0852c_row4_col43,#T_0852c_row4_col47,#T_0852c_row5_col0,#T_0852c_row5_col1,#T_0852c_row5_col2,#T_0852c_row5_col3,#T_0852c_row5_col4,#T_0852c_row5_col5,#T_0852c_row5_col6,#T_0852c_row5_col7,#T_0852c_row5_col8,#T_0852c_row5_col9,#T_0852c_row5_col10,#T_0852c_row5_col11,#T_0852c_row5_col15,#T_0852c_row5_col16,#T_0852c_row5_col17,#T_0852c_row5_col18,#T_0852c_row5_col19,#T_0852c_row5_col23,#T_0852c_row5_col24,#T_0852c_row5_col25,#T_0852c_row5_col26,#T_0852c_row5_col27,#T_0852c_row5_col31,#T_0852c_row5_col32,#T_0852c_row5_col33,#T_0852c_row5_col34,#T_0852c_row5_col35,#T_0852c_row5_col39{
            background-color:  green;
        }
</style>

<table id="T_0852c_" >
        <caption>Fixed points of the permutations. 🟩 = fixed, 🟥 = moved</caption>
        <thead>   
        <tr>        
        <th class="blank level0" ></th>        
        <th class="col_heading level0 col0" >0</th>        
        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>        <th class="col_heading level0 col18" >18</th>        <th class="col_heading level0 col19" >19</th>        <th class="col_heading level0 col20" >20</th>        <th class="col_heading level0 col21" >21</th>        <th class="col_heading level0 col22" >22</th>        <th class="col_heading level0 col23" >23</th>        <th class="col_heading level0 col24" >24</th>        <th class="col_heading level0 col25" >25</th>        <th class="col_heading level0 col26" >26</th>        <th class="col_heading level0 col27" >27</th>        <th class="col_heading level0 col28" >28</th>        <th class="col_heading level0 col29" >29</th>        <th class="col_heading level0 col30" >30</th>        <th class="col_heading level0 col31" >31</th>        <th class="col_heading level0 col32" >32</th>        <th class="col_heading level0 col33" >33</th>        <th class="col_heading level0 col34" >34</th>        <th class="col_heading level0 col35" >35</th>        <th class="col_heading level0 col36" >36</th>        <th class="col_heading level0 col37" >37</th>        <th class="col_heading level0 col38" >38</th>        <th class="col_heading level0 col39" >39</th>        <th class="col_heading level0 col40" >40</th>        <th class="col_heading level0 col41" >41</th>        <th class="col_heading level0 col42" >42</th>        <th class="col_heading level0 col43" >43</th>        <th class="col_heading level0 col44" >44</th>        <th class="col_heading level0 col45" >45</th>        <th class="col_heading level0 col46" >46</th>        <th class="col_heading level0 col47" >47</th>    </tr></thead>
        <tbody>
                <tr>
                        <th id="T_0852c_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_0852c_row0_col0" class="data row0 col0" >6</td>
                        <td id="T_0852c_row0_col1" class="data row0 col1" >7</td>
                        <td id="T_0852c_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_0852c_row0_col3" class="data row0 col3" >1</td>
                        <td id="T_0852c_row0_col4" class="data row0 col4" >2</td>
                        <td id="T_0852c_row0_col5" class="data row0 col5" >3</td>
                        <td id="T_0852c_row0_col6" class="data row0 col6" >4</td>
                        <td id="T_0852c_row0_col7" class="data row0 col7" >5</td>
                        <td id="T_0852c_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_0852c_row0_col9" class="data row0 col9" >17</td>
                        <td id="T_0852c_row0_col10" class="data row0 col10" >18</td>
                        <td id="T_0852c_row0_col11" class="data row0 col11" >11</td>
                        <td id="T_0852c_row0_col12" class="data row0 col12" >12</td>
                        <td id="T_0852c_row0_col13" class="data row0 col13" >13</td>
                        <td id="T_0852c_row0_col14" class="data row0 col14" >14</td>
                        <td id="T_0852c_row0_col15" class="data row0 col15" >15</td>
                        <td id="T_0852c_row0_col16" class="data row0 col16" >24</td>
                        <td id="T_0852c_row0_col17" class="data row0 col17" >25</td>
                        <td id="T_0852c_row0_col18" class="data row0 col18" >26</td>
                        <td id="T_0852c_row0_col19" class="data row0 col19" >19</td>
                        <td id="T_0852c_row0_col20" class="data row0 col20" >20</td>
                        <td id="T_0852c_row0_col21" class="data row0 col21" >21</td>
                        <td id="T_0852c_row0_col22" class="data row0 col22" >22</td>
                        <td id="T_0852c_row0_col23" class="data row0 col23" >23</td>
                        <td id="T_0852c_row0_col24" class="data row0 col24" >32</td>
                        <td id="T_0852c_row0_col25" class="data row0 col25" >33</td>
                        <td id="T_0852c_row0_col26" class="data row0 col26" >34</td>
                        <td id="T_0852c_row0_col27" class="data row0 col27" >27</td>
                        <td id="T_0852c_row0_col28" class="data row0 col28" >28</td>
                        <td id="T_0852c_row0_col29" class="data row0 col29" >29</td>
                        <td id="T_0852c_row0_col30" class="data row0 col30" >30</td>
                        <td id="T_0852c_row0_col31" class="data row0 col31" >31</td>
                        <td id="T_0852c_row0_col32" class="data row0 col32" >8</td>
                        <td id="T_0852c_row0_col33" class="data row0 col33" >9</td>
                        <td id="T_0852c_row0_col34" class="data row0 col34" >10</td>
                        <td id="T_0852c_row0_col35" class="data row0 col35" >35</td>
                        <td id="T_0852c_row0_col36" class="data row0 col36" >36</td>
                        <td id="T_0852c_row0_col37" class="data row0 col37" >37</td>
                        <td id="T_0852c_row0_col38" class="data row0 col38" >38</td>
                        <td id="T_0852c_row0_col39" class="data row0 col39" >39</td>
                        <td id="T_0852c_row0_col40" class="data row0 col40" >40</td>
                        <td id="T_0852c_row0_col41" class="data row0 col41" >41</td>
                        <td id="T_0852c_row0_col42" class="data row0 col42" >42</td>
                        <td id="T_0852c_row0_col43" class="data row0 col43" >43</td>
                        <td id="T_0852c_row0_col44" class="data row0 col44" >44</td>
                        <td id="T_0852c_row0_col45" class="data row0 col45" >45</td>
                        <td id="T_0852c_row0_col46" class="data row0 col46" >46</td>
                        <td id="T_0852c_row0_col47" class="data row0 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_0852c_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_0852c_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_0852c_row1_col1" class="data row1 col1" >1</td>
                        <td id="T_0852c_row1_col2" class="data row1 col2" >18</td>
                        <td id="T_0852c_row1_col3" class="data row1 col3" >19</td>
                        <td id="T_0852c_row1_col4" class="data row1 col4" >20</td>
                        <td id="T_0852c_row1_col5" class="data row1 col5" >5</td>
                        <td id="T_0852c_row1_col6" class="data row1 col6" >6</td>
                        <td id="T_0852c_row1_col7" class="data row1 col7" >7</td>
                        <td id="T_0852c_row1_col8" class="data row1 col8" >8</td>
                        <td id="T_0852c_row1_col9" class="data row1 col9" >9</td>
                        <td id="T_0852c_row1_col10" class="data row1 col10" >10</td>
                        <td id="T_0852c_row1_col11" class="data row1 col11" >11</td>
                        <td id="T_0852c_row1_col12" class="data row1 col12" >12</td>
                        <td id="T_0852c_row1_col13" class="data row1 col13" >13</td>
                        <td id="T_0852c_row1_col14" class="data row1 col14" >14</td>
                        <td id="T_0852c_row1_col15" class="data row1 col15" >15</td>
                        <td id="T_0852c_row1_col16" class="data row1 col16" >16</td>
                        <td id="T_0852c_row1_col17" class="data row1 col17" >17</td>
                        <td id="T_0852c_row1_col18" class="data row1 col18" >42</td>
                        <td id="T_0852c_row1_col19" class="data row1 col19" >43</td>
                        <td id="T_0852c_row1_col20" class="data row1 col20" >44</td>
                        <td id="T_0852c_row1_col21" class="data row1 col21" >21</td>
                        <td id="T_0852c_row1_col22" class="data row1 col22" >22</td>
                        <td id="T_0852c_row1_col23" class="data row1 col23" >23</td>
                        <td id="T_0852c_row1_col24" class="data row1 col24" >30</td>
                        <td id="T_0852c_row1_col25" class="data row1 col25" >31</td>
                        <td id="T_0852c_row1_col26" class="data row1 col26" >24</td>
                        <td id="T_0852c_row1_col27" class="data row1 col27" >25</td>
                        <td id="T_0852c_row1_col28" class="data row1 col28" >26</td>
                        <td id="T_0852c_row1_col29" class="data row1 col29" >27</td>
                        <td id="T_0852c_row1_col30" class="data row1 col30" >28</td>
                        <td id="T_0852c_row1_col31" class="data row1 col31" >29</td>
                        <td id="T_0852c_row1_col32" class="data row1 col32" >4</td>
                        <td id="T_0852c_row1_col33" class="data row1 col33" >33</td>
                        <td id="T_0852c_row1_col34" class="data row1 col34" >34</td>
                        <td id="T_0852c_row1_col35" class="data row1 col35" >35</td>
                        <td id="T_0852c_row1_col36" class="data row1 col36" >36</td>
                        <td id="T_0852c_row1_col37" class="data row1 col37" >37</td>
                        <td id="T_0852c_row1_col38" class="data row1 col38" >2</td>
                        <td id="T_0852c_row1_col39" class="data row1 col39" >3</td>
                        <td id="T_0852c_row1_col40" class="data row1 col40" >40</td>
                        <td id="T_0852c_row1_col41" class="data row1 col41" >41</td>
                        <td id="T_0852c_row1_col42" class="data row1 col42" >38</td>
                        <td id="T_0852c_row1_col43" class="data row1 col43" >39</td>
                        <td id="T_0852c_row1_col44" class="data row1 col44" >32</td>
                        <td id="T_0852c_row1_col45" class="data row1 col45" >45</td>
                        <td id="T_0852c_row1_col46" class="data row1 col46" >46</td>
                        <td id="T_0852c_row1_col47" class="data row1 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_0852c_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_0852c_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_0852c_row2_col1" class="data row2 col1" >1</td>
                        <td id="T_0852c_row2_col2" class="data row2 col2" >2</td>
                        <td id="T_0852c_row2_col3" class="data row2 col3" >3</td>
                        <td id="T_0852c_row2_col4" class="data row2 col4" >10</td>
                        <td id="T_0852c_row2_col5" class="data row2 col5" >11</td>
                        <td id="T_0852c_row2_col6" class="data row2 col6" >12</td>
                        <td id="T_0852c_row2_col7" class="data row2 col7" >7</td>
                        <td id="T_0852c_row2_col8" class="data row2 col8" >8</td>
                        <td id="T_0852c_row2_col9" class="data row2 col9" >9</td>
                        <td id="T_0852c_row2_col10" class="data row2 col10" >40</td>
                        <td id="T_0852c_row2_col11" class="data row2 col11" >41</td>
                        <td id="T_0852c_row2_col12" class="data row2 col12" >42</td>
                        <td id="T_0852c_row2_col13" class="data row2 col13" >13</td>
                        <td id="T_0852c_row2_col14" class="data row2 col14" >14</td>
                        <td id="T_0852c_row2_col15" class="data row2 col15" >15</td>
                        <td id="T_0852c_row2_col16" class="data row2 col16" >22</td>
                        <td id="T_0852c_row2_col17" class="data row2 col17" >23</td>
                        <td id="T_0852c_row2_col18" class="data row2 col18" >16</td>
                        <td id="T_0852c_row2_col19" class="data row2 col19" >17</td>
                        <td id="T_0852c_row2_col20" class="data row2 col20" >18</td>
                        <td id="T_0852c_row2_col21" class="data row2 col21" >19</td>
                        <td id="T_0852c_row2_col22" class="data row2 col22" >20</td>
                        <td id="T_0852c_row2_col23" class="data row2 col23" >21</td>
                        <td id="T_0852c_row2_col24" class="data row2 col24" >6</td>
                        <td id="T_0852c_row2_col25" class="data row2 col25" >25</td>
                        <td id="T_0852c_row2_col26" class="data row2 col26" >26</td>
                        <td id="T_0852c_row2_col27" class="data row2 col27" >27</td>
                        <td id="T_0852c_row2_col28" class="data row2 col28" >28</td>
                        <td id="T_0852c_row2_col29" class="data row2 col29" >29</td>
                        <td id="T_0852c_row2_col30" class="data row2 col30" >4</td>
                        <td id="T_0852c_row2_col31" class="data row2 col31" >5</td>
                        <td id="T_0852c_row2_col32" class="data row2 col32" >32</td>
                        <td id="T_0852c_row2_col33" class="data row2 col33" >33</td>
                        <td id="T_0852c_row2_col34" class="data row2 col34" >34</td>
                        <td id="T_0852c_row2_col35" class="data row2 col35" >35</td>
                        <td id="T_0852c_row2_col36" class="data row2 col36" >36</td>
                        <td id="T_0852c_row2_col37" class="data row2 col37" >37</td>
                        <td id="T_0852c_row2_col38" class="data row2 col38" >38</td>
                        <td id="T_0852c_row2_col39" class="data row2 col39" >39</td>
                        <td id="T_0852c_row2_col40" class="data row2 col40" >30</td>
                        <td id="T_0852c_row2_col41" class="data row2 col41" >31</td>
                        <td id="T_0852c_row2_col42" class="data row2 col42" >24</td>
                        <td id="T_0852c_row2_col43" class="data row2 col43" >43</td>
                        <td id="T_0852c_row2_col44" class="data row2 col44" >44</td>
                        <td id="T_0852c_row2_col45" class="data row2 col45" >45</td>
                        <td id="T_0852c_row2_col46" class="data row2 col46" >46</td>
                        <td id="T_0852c_row2_col47" class="data row2 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_0852c_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_0852c_row3_col0" class="data row3 col0" >36</td>
                        <td id="T_0852c_row3_col1" class="data row3 col1" >1</td>
                        <td id="T_0852c_row3_col2" class="data row3 col2" >2</td>
                        <td id="T_0852c_row3_col3" class="data row3 col3" >3</td>
                        <td id="T_0852c_row3_col4" class="data row3 col4" >4</td>
                        <td id="T_0852c_row3_col5" class="data row3 col5" >5</td>
                        <td id="T_0852c_row3_col6" class="data row3 col6" >34</td>
                        <td id="T_0852c_row3_col7" class="data row3 col7" >35</td>
                        <td id="T_0852c_row3_col8" class="data row3 col8" >14</td>
                        <td id="T_0852c_row3_col9" class="data row3 col9" >15</td>
                        <td id="T_0852c_row3_col10" class="data row3 col10" >8</td>
                        <td id="T_0852c_row3_col11" class="data row3 col11" >9</td>
                        <td id="T_0852c_row3_col12" class="data row3 col12" >10</td>
                        <td id="T_0852c_row3_col13" class="data row3 col13" >11</td>
                        <td id="T_0852c_row3_col14" class="data row3 col14" >12</td>
                        <td id="T_0852c_row3_col15" class="data row3 col15" >13</td>
                        <td id="T_0852c_row3_col16" class="data row3 col16" >0</td>
                        <td id="T_0852c_row3_col17" class="data row3 col17" >17</td>
                        <td id="T_0852c_row3_col18" class="data row3 col18" >18</td>
                        <td id="T_0852c_row3_col19" class="data row3 col19" >19</td>
                        <td id="T_0852c_row3_col20" class="data row3 col20" >20</td>
                        <td id="T_0852c_row3_col21" class="data row3 col21" >21</td>
                        <td id="T_0852c_row3_col22" class="data row3 col22" >6</td>
                        <td id="T_0852c_row3_col23" class="data row3 col23" >7</td>
                        <td id="T_0852c_row3_col24" class="data row3 col24" >24</td>
                        <td id="T_0852c_row3_col25" class="data row3 col25" >25</td>
                        <td id="T_0852c_row3_col26" class="data row3 col26" >26</td>
                        <td id="T_0852c_row3_col27" class="data row3 col27" >27</td>
                        <td id="T_0852c_row3_col28" class="data row3 col28" >28</td>
                        <td id="T_0852c_row3_col29" class="data row3 col29" >29</td>
                        <td id="T_0852c_row3_col30" class="data row3 col30" >30</td>
                        <td id="T_0852c_row3_col31" class="data row3 col31" >31</td>
                        <td id="T_0852c_row3_col32" class="data row3 col32" >32</td>
                        <td id="T_0852c_row3_col33" class="data row3 col33" >33</td>
                        <td id="T_0852c_row3_col34" class="data row3 col34" >46</td>
                        <td id="T_0852c_row3_col35" class="data row3 col35" >47</td>
                        <td id="T_0852c_row3_col36" class="data row3 col36" >40</td>
                        <td id="T_0852c_row3_col37" class="data row3 col37" >37</td>
                        <td id="T_0852c_row3_col38" class="data row3 col38" >38</td>
                        <td id="T_0852c_row3_col39" class="data row3 col39" >39</td>
                        <td id="T_0852c_row3_col40" class="data row3 col40" >16</td>
                        <td id="T_0852c_row3_col41" class="data row3 col41" >41</td>
                        <td id="T_0852c_row3_col42" class="data row3 col42" >42</td>
                        <td id="T_0852c_row3_col43" class="data row3 col43" >43</td>
                        <td id="T_0852c_row3_col44" class="data row3 col44" >44</td>
                        <td id="T_0852c_row3_col45" class="data row3 col45" >45</td>
                        <td id="T_0852c_row3_col46" class="data row3 col46" >22</td>
                        <td id="T_0852c_row3_col47" class="data row3 col47" >23</td>
            </tr>
            <tr>
                        <th id="T_0852c_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_0852c_row4_col0" class="data row4 col0" >26</td>
                        <td id="T_0852c_row4_col1" class="data row4 col1" >27</td>
                        <td id="T_0852c_row4_col2" class="data row4 col2" >28</td>
                        <td id="T_0852c_row4_col3" class="data row4 col3" >3</td>
                        <td id="T_0852c_row4_col4" class="data row4 col4" >4</td>
                        <td id="T_0852c_row4_col5" class="data row4 col5" >5</td>
                        <td id="T_0852c_row4_col6" class="data row4 col6" >6</td>
                        <td id="T_0852c_row4_col7" class="data row4 col7" >7</td>
                        <td id="T_0852c_row4_col8" class="data row4 col8" >2</td>
                        <td id="T_0852c_row4_col9" class="data row4 col9" >9</td>
                        <td id="T_0852c_row4_col10" class="data row4 col10" >10</td>
                        <td id="T_0852c_row4_col11" class="data row4 col11" >11</td>
                        <td id="T_0852c_row4_col12" class="data row4 col12" >12</td>
                        <td id="T_0852c_row4_col13" class="data row4 col13" >13</td>
                        <td id="T_0852c_row4_col14" class="data row4 col14" >0</td>
                        <td id="T_0852c_row4_col15" class="data row4 col15" >1</td>
                        <td id="T_0852c_row4_col16" class="data row4 col16" >16</td>
                        <td id="T_0852c_row4_col17" class="data row4 col17" >17</td>
                        <td id="T_0852c_row4_col18" class="data row4 col18" >18</td>
                        <td id="T_0852c_row4_col19" class="data row4 col19" >19</td>
                        <td id="T_0852c_row4_col20" class="data row4 col20" >20</td>
                        <td id="T_0852c_row4_col21" class="data row4 col21" >21</td>
                        <td id="T_0852c_row4_col22" class="data row4 col22" >22</td>
                        <td id="T_0852c_row4_col23" class="data row4 col23" >23</td>
                        <td id="T_0852c_row4_col24" class="data row4 col24" >24</td>
                        <td id="T_0852c_row4_col25" class="data row4 col25" >25</td>
                        <td id="T_0852c_row4_col26" class="data row4 col26" >44</td>
                        <td id="T_0852c_row4_col27" class="data row4 col27" >45</td>
                        <td id="T_0852c_row4_col28" class="data row4 col28" >46</td>
                        <td id="T_0852c_row4_col29" class="data row4 col29" >29</td>
                        <td id="T_0852c_row4_col30" class="data row4 col30" >30</td>
                        <td id="T_0852c_row4_col31" class="data row4 col31" >31</td>
                        <td id="T_0852c_row4_col32" class="data row4 col32" >38</td>
                        <td id="T_0852c_row4_col33" class="data row4 col33" >39</td>
                        <td id="T_0852c_row4_col34" class="data row4 col34" >32</td>
                        <td id="T_0852c_row4_col35" class="data row4 col35" >33</td>
                        <td id="T_0852c_row4_col36" class="data row4 col36" >34</td>
                        <td id="T_0852c_row4_col37" class="data row4 col37" >35</td>
                        <td id="T_0852c_row4_col38" class="data row4 col38" >36</td>
                        <td id="T_0852c_row4_col39" class="data row4 col39" >37</td>
                        <td id="T_0852c_row4_col40" class="data row4 col40" >40</td>
                        <td id="T_0852c_row4_col41" class="data row4 col41" >41</td>
                        <td id="T_0852c_row4_col42" class="data row4 col42" >42</td>
                        <td id="T_0852c_row4_col43" class="data row4 col43" >43</td>
                        <td id="T_0852c_row4_col44" class="data row4 col44" >14</td>
                        <td id="T_0852c_row4_col45" class="data row4 col45" >15</td>
                        <td id="T_0852c_row4_col46" class="data row4 col46" >8</td>
                        <td id="T_0852c_row4_col47" class="data row4 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_0852c_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_0852c_row5_col0" class="data row5 col0" >0</td>
                        <td id="T_0852c_row5_col1" class="data row5 col1" >1</td>
                        <td id="T_0852c_row5_col2" class="data row5 col2" >2</td>
                        <td id="T_0852c_row5_col3" class="data row5 col3" >3</td>
                        <td id="T_0852c_row5_col4" class="data row5 col4" >4</td>
                        <td id="T_0852c_row5_col5" class="data row5 col5" >5</td>
                        <td id="T_0852c_row5_col6" class="data row5 col6" >6</td>
                        <td id="T_0852c_row5_col7" class="data row5 col7" >7</td>
                        <td id="T_0852c_row5_col8" class="data row5 col8" >8</td>
                        <td id="T_0852c_row5_col9" class="data row5 col9" >9</td>
                        <td id="T_0852c_row5_col10" class="data row5 col10" >10</td>
                        <td id="T_0852c_row5_col11" class="data row5 col11" >11</td>
                        <td id="T_0852c_row5_col12" class="data row5 col12" >36</td>
                        <td id="T_0852c_row5_col13" class="data row5 col13" >37</td>
                        <td id="T_0852c_row5_col14" class="data row5 col14" >38</td>
                        <td id="T_0852c_row5_col15" class="data row5 col15" >15</td>
                        <td id="T_0852c_row5_col16" class="data row5 col16" >16</td>
                        <td id="T_0852c_row5_col17" class="data row5 col17" >17</td>
                        <td id="T_0852c_row5_col18" class="data row5 col18" >18</td>
                        <td id="T_0852c_row5_col19" class="data row5 col19" >19</td>
                        <td id="T_0852c_row5_col20" class="data row5 col20" >12</td>
                        <td id="T_0852c_row5_col21" class="data row5 col21" >13</td>
                        <td id="T_0852c_row5_col22" class="data row5 col22" >14</td>
                        <td id="T_0852c_row5_col23" class="data row5 col23" >23</td>
                        <td id="T_0852c_row5_col24" class="data row5 col24" >24</td>
                        <td id="T_0852c_row5_col25" class="data row5 col25" >25</td>
                        <td id="T_0852c_row5_col26" class="data row5 col26" >26</td>
                        <td id="T_0852c_row5_col27" class="data row5 col27" >27</td>
                        <td id="T_0852c_row5_col28" class="data row5 col28" >20</td>
                        <td id="T_0852c_row5_col29" class="data row5 col29" >21</td>
                        <td id="T_0852c_row5_col30" class="data row5 col30" >22</td>
                        <td id="T_0852c_row5_col31" class="data row5 col31" >31</td>
                        <td id="T_0852c_row5_col32" class="data row5 col32" >32</td>
                        <td id="T_0852c_row5_col33" class="data row5 col33" >33</td>
                        <td id="T_0852c_row5_col34" class="data row5 col34" >34</td>
                        <td id="T_0852c_row5_col35" class="data row5 col35" >35</td>
                        <td id="T_0852c_row5_col36" class="data row5 col36" >28</td>
                        <td id="T_0852c_row5_col37" class="data row5 col37" >29</td>
                        <td id="T_0852c_row5_col38" class="data row5 col38" >30</td>
                        <td id="T_0852c_row5_col39" class="data row5 col39" >39</td>
                        <td id="T_0852c_row5_col40" class="data row5 col40" >46</td>
                        <td id="T_0852c_row5_col41" class="data row5 col41" >47</td>
                        <td id="T_0852c_row5_col42" class="data row5 col42" >40</td>
                        <td id="T_0852c_row5_col43" class="data row5 col43" >41</td>
                        <td id="T_0852c_row5_col44" class="data row5 col44" >42</td>
                        <td id="T_0852c_row5_col45" class="data row5 col45" >43</td>
                        <td id="T_0852c_row5_col46" class="data row5 col46" >44</td>
                        <td id="T_0852c_row5_col47" class="data row5 col47" >45</td>
            </tr>
    </tbody></table>
</div>


We already knew that each permutation only affects 20 items, but there does seem to be quite a bit more structure that we need to explore.

1. Permutations affect big blocks consecutive elements, and leave other large blocks intact.
2. Each row has one big block which is permuted, and several other tinier blocks of size < 4.
3. Most blocks (tiny and large) seem to preserve some order.
4. Groups of 8 columns seem to behave in a very structured way.

These observations by themselves are not so helpful, but together they make solution incredibly obvious (with hindsight).
What finally made it click was when we used colors to represent each group of 8 columns. 

In the following plot, we assigned to each element a color depending on its value divided by 8. 
We also chose to represent the elements which 


```python
COLORS = {
    0: 'white', # ⬜️ UP
    1: 'orange', # 🟧 RIGHT
    2: 'green', # 🟩 FRONT
    3: 'red', # 🟥 LEFT
    4: 'blue', # 🟦 BACK
    5: 'yellow', # 🟨 DOWN
    -1: 'magenta' # 🟪
}

def color_group(val):
    return f'background-color: {COLORS[val//BLOCK_SIZE]}' 

def color_not_id_in_group(row):
    return ['color: magenta' if row[i] != i and row[i] // BLOCK_SIZE == i // BLOCK_SIZE else '' for i in range(len(row))]

PERMUTATIONS_DF.style.set_caption('Permutations visualized with "arbitrary" colors. 🟪 = move to same block').applymap(color_group).apply(color_not_id_in_group,axis=1)
```




<style  type="text/css" >
#T_06b0c_row0_col0,#T_06b0c_row0_col1,#T_06b0c_row0_col2,#T_06b0c_row0_col3,#T_06b0c_row0_col4,#T_06b0c_row0_col5,#T_06b0c_row0_col6,#T_06b0c_row0_col7{
            background-color:  white;
            color:  magenta;
        }#T_06b0c_row0_col8,#T_06b0c_row0_col9,#T_06b0c_row0_col10,#T_06b0c_row0_col19,#T_06b0c_row0_col20,#T_06b0c_row0_col21,#T_06b0c_row0_col22,#T_06b0c_row0_col23,#T_06b0c_row1_col2,#T_06b0c_row1_col3,#T_06b0c_row1_col4,#T_06b0c_row1_col16,#T_06b0c_row1_col17,#T_06b0c_row1_col21,#T_06b0c_row1_col22,#T_06b0c_row1_col23,#T_06b0c_row3_col17,#T_06b0c_row3_col18,#T_06b0c_row3_col19,#T_06b0c_row3_col20,#T_06b0c_row3_col21,#T_06b0c_row3_col40,#T_06b0c_row3_col46,#T_06b0c_row3_col47,#T_06b0c_row4_col16,#T_06b0c_row4_col17,#T_06b0c_row4_col18,#T_06b0c_row4_col19,#T_06b0c_row4_col20,#T_06b0c_row4_col21,#T_06b0c_row4_col22,#T_06b0c_row4_col23,#T_06b0c_row5_col16,#T_06b0c_row5_col17,#T_06b0c_row5_col18,#T_06b0c_row5_col19,#T_06b0c_row5_col23,#T_06b0c_row5_col28,#T_06b0c_row5_col29,#T_06b0c_row5_col30{
            background-color:  green;
        }#T_06b0c_row0_col11,#T_06b0c_row0_col12,#T_06b0c_row0_col13,#T_06b0c_row0_col14,#T_06b0c_row0_col15,#T_06b0c_row0_col32,#T_06b0c_row0_col33,#T_06b0c_row0_col34,#T_06b0c_row1_col8,#T_06b0c_row1_col9,#T_06b0c_row1_col10,#T_06b0c_row1_col11,#T_06b0c_row1_col12,#T_06b0c_row1_col13,#T_06b0c_row1_col14,#T_06b0c_row1_col15,#T_06b0c_row2_col4,#T_06b0c_row2_col5,#T_06b0c_row2_col6,#T_06b0c_row2_col8,#T_06b0c_row2_col9,#T_06b0c_row2_col13,#T_06b0c_row2_col14,#T_06b0c_row2_col15,#T_06b0c_row4_col9,#T_06b0c_row4_col10,#T_06b0c_row4_col11,#T_06b0c_row4_col12,#T_06b0c_row4_col13,#T_06b0c_row4_col44,#T_06b0c_row4_col45,#T_06b0c_row4_col46,#T_06b0c_row5_col8,#T_06b0c_row5_col9,#T_06b0c_row5_col10,#T_06b0c_row5_col11,#T_06b0c_row5_col15,#T_06b0c_row5_col20,#T_06b0c_row5_col21,#T_06b0c_row5_col22{
            background-color:  orange;
        }#T_06b0c_row0_col16,#T_06b0c_row0_col17,#T_06b0c_row0_col18,#T_06b0c_row0_col27,#T_06b0c_row0_col28,#T_06b0c_row0_col29,#T_06b0c_row0_col30,#T_06b0c_row0_col31,#T_06b0c_row2_col25,#T_06b0c_row2_col26,#T_06b0c_row2_col27,#T_06b0c_row2_col28,#T_06b0c_row2_col29,#T_06b0c_row2_col40,#T_06b0c_row2_col41,#T_06b0c_row2_col42,#T_06b0c_row3_col24,#T_06b0c_row3_col25,#T_06b0c_row3_col26,#T_06b0c_row3_col27,#T_06b0c_row3_col28,#T_06b0c_row3_col29,#T_06b0c_row3_col30,#T_06b0c_row3_col31,#T_06b0c_row4_col0,#T_06b0c_row4_col1,#T_06b0c_row4_col2,#T_06b0c_row4_col24,#T_06b0c_row4_col25,#T_06b0c_row4_col29,#T_06b0c_row4_col30,#T_06b0c_row4_col31,#T_06b0c_row5_col24,#T_06b0c_row5_col25,#T_06b0c_row5_col26,#T_06b0c_row5_col27,#T_06b0c_row5_col31,#T_06b0c_row5_col36,#T_06b0c_row5_col37,#T_06b0c_row5_col38{
            background-color:  red;
        }#T_06b0c_row0_col24,#T_06b0c_row0_col25,#T_06b0c_row0_col26,#T_06b0c_row0_col35,#T_06b0c_row0_col36,#T_06b0c_row0_col37,#T_06b0c_row0_col38,#T_06b0c_row0_col39,#T_06b0c_row1_col33,#T_06b0c_row1_col34,#T_06b0c_row1_col35,#T_06b0c_row1_col36,#T_06b0c_row1_col37,#T_06b0c_row1_col42,#T_06b0c_row1_col43,#T_06b0c_row1_col44,#T_06b0c_row2_col32,#T_06b0c_row2_col33,#T_06b0c_row2_col34,#T_06b0c_row2_col35,#T_06b0c_row2_col36,#T_06b0c_row2_col37,#T_06b0c_row2_col38,#T_06b0c_row2_col39,#T_06b0c_row3_col0,#T_06b0c_row3_col6,#T_06b0c_row3_col7,#T_06b0c_row3_col32,#T_06b0c_row3_col33,#T_06b0c_row3_col37,#T_06b0c_row3_col38,#T_06b0c_row3_col39,#T_06b0c_row5_col12,#T_06b0c_row5_col13,#T_06b0c_row5_col14,#T_06b0c_row5_col32,#T_06b0c_row5_col33,#T_06b0c_row5_col34,#T_06b0c_row5_col35,#T_06b0c_row5_col39{
            background-color:  blue;
        }#T_06b0c_row0_col40,#T_06b0c_row0_col41,#T_06b0c_row0_col42,#T_06b0c_row0_col43,#T_06b0c_row0_col44,#T_06b0c_row0_col45,#T_06b0c_row0_col46,#T_06b0c_row0_col47,#T_06b0c_row1_col18,#T_06b0c_row1_col19,#T_06b0c_row1_col20,#T_06b0c_row1_col40,#T_06b0c_row1_col41,#T_06b0c_row1_col45,#T_06b0c_row1_col46,#T_06b0c_row1_col47,#T_06b0c_row2_col10,#T_06b0c_row2_col11,#T_06b0c_row2_col12,#T_06b0c_row2_col43,#T_06b0c_row2_col44,#T_06b0c_row2_col45,#T_06b0c_row2_col46,#T_06b0c_row2_col47,#T_06b0c_row3_col34,#T_06b0c_row3_col35,#T_06b0c_row3_col36,#T_06b0c_row3_col41,#T_06b0c_row3_col42,#T_06b0c_row3_col43,#T_06b0c_row3_col44,#T_06b0c_row3_col45,#T_06b0c_row4_col26,#T_06b0c_row4_col27,#T_06b0c_row4_col28,#T_06b0c_row4_col40,#T_06b0c_row4_col41,#T_06b0c_row4_col42,#T_06b0c_row4_col43,#T_06b0c_row4_col47{
            background-color:  yellow;
        }#T_06b0c_row1_col0,#T_06b0c_row1_col1,#T_06b0c_row1_col5,#T_06b0c_row1_col6,#T_06b0c_row1_col7,#T_06b0c_row1_col32,#T_06b0c_row1_col38,#T_06b0c_row1_col39,#T_06b0c_row2_col0,#T_06b0c_row2_col1,#T_06b0c_row2_col2,#T_06b0c_row2_col3,#T_06b0c_row2_col7,#T_06b0c_row2_col24,#T_06b0c_row2_col30,#T_06b0c_row2_col31,#T_06b0c_row3_col1,#T_06b0c_row3_col2,#T_06b0c_row3_col3,#T_06b0c_row3_col4,#T_06b0c_row3_col5,#T_06b0c_row3_col16,#T_06b0c_row3_col22,#T_06b0c_row3_col23,#T_06b0c_row4_col3,#T_06b0c_row4_col4,#T_06b0c_row4_col5,#T_06b0c_row4_col6,#T_06b0c_row4_col7,#T_06b0c_row4_col8,#T_06b0c_row4_col14,#T_06b0c_row4_col15,#T_06b0c_row5_col0,#T_06b0c_row5_col1,#T_06b0c_row5_col2,#T_06b0c_row5_col3,#T_06b0c_row5_col4,#T_06b0c_row5_col5,#T_06b0c_row5_col6,#T_06b0c_row5_col7{
            background-color:  white;
        }#T_06b0c_row1_col24,#T_06b0c_row1_col25,#T_06b0c_row1_col26,#T_06b0c_row1_col27,#T_06b0c_row1_col28,#T_06b0c_row1_col29,#T_06b0c_row1_col30,#T_06b0c_row1_col31{
            background-color:  red;
            color:  magenta;
        }#T_06b0c_row2_col16,#T_06b0c_row2_col17,#T_06b0c_row2_col18,#T_06b0c_row2_col19,#T_06b0c_row2_col20,#T_06b0c_row2_col21,#T_06b0c_row2_col22,#T_06b0c_row2_col23{
            background-color:  green;
            color:  magenta;
        }#T_06b0c_row3_col8,#T_06b0c_row3_col9,#T_06b0c_row3_col10,#T_06b0c_row3_col11,#T_06b0c_row3_col12,#T_06b0c_row3_col13,#T_06b0c_row3_col14,#T_06b0c_row3_col15{
            background-color:  orange;
            color:  magenta;
        }#T_06b0c_row4_col32,#T_06b0c_row4_col33,#T_06b0c_row4_col34,#T_06b0c_row4_col35,#T_06b0c_row4_col36,#T_06b0c_row4_col37,#T_06b0c_row4_col38,#T_06b0c_row4_col39{
            background-color:  blue;
            color:  magenta;
        }#T_06b0c_row5_col40,#T_06b0c_row5_col41,#T_06b0c_row5_col42,#T_06b0c_row5_col43,#T_06b0c_row5_col44,#T_06b0c_row5_col45,#T_06b0c_row5_col46,#T_06b0c_row5_col47{
            background-color:  yellow;
            color:  magenta;
        }</style>
<table id="T_06b0c_" ><caption>Permutations visualized with "arbitrary" colors. 🟪 = move to same block</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>        <th class="col_heading level0 col18" >18</th>        <th class="col_heading level0 col19" >19</th>        <th class="col_heading level0 col20" >20</th>        <th class="col_heading level0 col21" >21</th>        <th class="col_heading level0 col22" >22</th>        <th class="col_heading level0 col23" >23</th>        <th class="col_heading level0 col24" >24</th>        <th class="col_heading level0 col25" >25</th>        <th class="col_heading level0 col26" >26</th>        <th class="col_heading level0 col27" >27</th>        <th class="col_heading level0 col28" >28</th>        <th class="col_heading level0 col29" >29</th>        <th class="col_heading level0 col30" >30</th>        <th class="col_heading level0 col31" >31</th>        <th class="col_heading level0 col32" >32</th>        <th class="col_heading level0 col33" >33</th>        <th class="col_heading level0 col34" >34</th>        <th class="col_heading level0 col35" >35</th>        <th class="col_heading level0 col36" >36</th>        <th class="col_heading level0 col37" >37</th>        <th class="col_heading level0 col38" >38</th>        <th class="col_heading level0 col39" >39</th>        <th class="col_heading level0 col40" >40</th>        <th class="col_heading level0 col41" >41</th>        <th class="col_heading level0 col42" >42</th>        <th class="col_heading level0 col43" >43</th>        <th class="col_heading level0 col44" >44</th>        <th class="col_heading level0 col45" >45</th>        <th class="col_heading level0 col46" >46</th>        <th class="col_heading level0 col47" >47</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_06b0c_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_06b0c_row0_col0" class="data row0 col0" >6</td>
                        <td id="T_06b0c_row0_col1" class="data row0 col1" >7</td>
                        <td id="T_06b0c_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_06b0c_row0_col3" class="data row0 col3" >1</td>
                        <td id="T_06b0c_row0_col4" class="data row0 col4" >2</td>
                        <td id="T_06b0c_row0_col5" class="data row0 col5" >3</td>
                        <td id="T_06b0c_row0_col6" class="data row0 col6" >4</td>
                        <td id="T_06b0c_row0_col7" class="data row0 col7" >5</td>
                        <td id="T_06b0c_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_06b0c_row0_col9" class="data row0 col9" >17</td>
                        <td id="T_06b0c_row0_col10" class="data row0 col10" >18</td>
                        <td id="T_06b0c_row0_col11" class="data row0 col11" >11</td>
                        <td id="T_06b0c_row0_col12" class="data row0 col12" >12</td>
                        <td id="T_06b0c_row0_col13" class="data row0 col13" >13</td>
                        <td id="T_06b0c_row0_col14" class="data row0 col14" >14</td>
                        <td id="T_06b0c_row0_col15" class="data row0 col15" >15</td>
                        <td id="T_06b0c_row0_col16" class="data row0 col16" >24</td>
                        <td id="T_06b0c_row0_col17" class="data row0 col17" >25</td>
                        <td id="T_06b0c_row0_col18" class="data row0 col18" >26</td>
                        <td id="T_06b0c_row0_col19" class="data row0 col19" >19</td>
                        <td id="T_06b0c_row0_col20" class="data row0 col20" >20</td>
                        <td id="T_06b0c_row0_col21" class="data row0 col21" >21</td>
                        <td id="T_06b0c_row0_col22" class="data row0 col22" >22</td>
                        <td id="T_06b0c_row0_col23" class="data row0 col23" >23</td>
                        <td id="T_06b0c_row0_col24" class="data row0 col24" >32</td>
                        <td id="T_06b0c_row0_col25" class="data row0 col25" >33</td>
                        <td id="T_06b0c_row0_col26" class="data row0 col26" >34</td>
                        <td id="T_06b0c_row0_col27" class="data row0 col27" >27</td>
                        <td id="T_06b0c_row0_col28" class="data row0 col28" >28</td>
                        <td id="T_06b0c_row0_col29" class="data row0 col29" >29</td>
                        <td id="T_06b0c_row0_col30" class="data row0 col30" >30</td>
                        <td id="T_06b0c_row0_col31" class="data row0 col31" >31</td>
                        <td id="T_06b0c_row0_col32" class="data row0 col32" >8</td>
                        <td id="T_06b0c_row0_col33" class="data row0 col33" >9</td>
                        <td id="T_06b0c_row0_col34" class="data row0 col34" >10</td>
                        <td id="T_06b0c_row0_col35" class="data row0 col35" >35</td>
                        <td id="T_06b0c_row0_col36" class="data row0 col36" >36</td>
                        <td id="T_06b0c_row0_col37" class="data row0 col37" >37</td>
                        <td id="T_06b0c_row0_col38" class="data row0 col38" >38</td>
                        <td id="T_06b0c_row0_col39" class="data row0 col39" >39</td>
                        <td id="T_06b0c_row0_col40" class="data row0 col40" >40</td>
                        <td id="T_06b0c_row0_col41" class="data row0 col41" >41</td>
                        <td id="T_06b0c_row0_col42" class="data row0 col42" >42</td>
                        <td id="T_06b0c_row0_col43" class="data row0 col43" >43</td>
                        <td id="T_06b0c_row0_col44" class="data row0 col44" >44</td>
                        <td id="T_06b0c_row0_col45" class="data row0 col45" >45</td>
                        <td id="T_06b0c_row0_col46" class="data row0 col46" >46</td>
                        <td id="T_06b0c_row0_col47" class="data row0 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_06b0c_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_06b0c_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_06b0c_row1_col1" class="data row1 col1" >1</td>
                        <td id="T_06b0c_row1_col2" class="data row1 col2" >18</td>
                        <td id="T_06b0c_row1_col3" class="data row1 col3" >19</td>
                        <td id="T_06b0c_row1_col4" class="data row1 col4" >20</td>
                        <td id="T_06b0c_row1_col5" class="data row1 col5" >5</td>
                        <td id="T_06b0c_row1_col6" class="data row1 col6" >6</td>
                        <td id="T_06b0c_row1_col7" class="data row1 col7" >7</td>
                        <td id="T_06b0c_row1_col8" class="data row1 col8" >8</td>
                        <td id="T_06b0c_row1_col9" class="data row1 col9" >9</td>
                        <td id="T_06b0c_row1_col10" class="data row1 col10" >10</td>
                        <td id="T_06b0c_row1_col11" class="data row1 col11" >11</td>
                        <td id="T_06b0c_row1_col12" class="data row1 col12" >12</td>
                        <td id="T_06b0c_row1_col13" class="data row1 col13" >13</td>
                        <td id="T_06b0c_row1_col14" class="data row1 col14" >14</td>
                        <td id="T_06b0c_row1_col15" class="data row1 col15" >15</td>
                        <td id="T_06b0c_row1_col16" class="data row1 col16" >16</td>
                        <td id="T_06b0c_row1_col17" class="data row1 col17" >17</td>
                        <td id="T_06b0c_row1_col18" class="data row1 col18" >42</td>
                        <td id="T_06b0c_row1_col19" class="data row1 col19" >43</td>
                        <td id="T_06b0c_row1_col20" class="data row1 col20" >44</td>
                        <td id="T_06b0c_row1_col21" class="data row1 col21" >21</td>
                        <td id="T_06b0c_row1_col22" class="data row1 col22" >22</td>
                        <td id="T_06b0c_row1_col23" class="data row1 col23" >23</td>
                        <td id="T_06b0c_row1_col24" class="data row1 col24" >30</td>
                        <td id="T_06b0c_row1_col25" class="data row1 col25" >31</td>
                        <td id="T_06b0c_row1_col26" class="data row1 col26" >24</td>
                        <td id="T_06b0c_row1_col27" class="data row1 col27" >25</td>
                        <td id="T_06b0c_row1_col28" class="data row1 col28" >26</td>
                        <td id="T_06b0c_row1_col29" class="data row1 col29" >27</td>
                        <td id="T_06b0c_row1_col30" class="data row1 col30" >28</td>
                        <td id="T_06b0c_row1_col31" class="data row1 col31" >29</td>
                        <td id="T_06b0c_row1_col32" class="data row1 col32" >4</td>
                        <td id="T_06b0c_row1_col33" class="data row1 col33" >33</td>
                        <td id="T_06b0c_row1_col34" class="data row1 col34" >34</td>
                        <td id="T_06b0c_row1_col35" class="data row1 col35" >35</td>
                        <td id="T_06b0c_row1_col36" class="data row1 col36" >36</td>
                        <td id="T_06b0c_row1_col37" class="data row1 col37" >37</td>
                        <td id="T_06b0c_row1_col38" class="data row1 col38" >2</td>
                        <td id="T_06b0c_row1_col39" class="data row1 col39" >3</td>
                        <td id="T_06b0c_row1_col40" class="data row1 col40" >40</td>
                        <td id="T_06b0c_row1_col41" class="data row1 col41" >41</td>
                        <td id="T_06b0c_row1_col42" class="data row1 col42" >38</td>
                        <td id="T_06b0c_row1_col43" class="data row1 col43" >39</td>
                        <td id="T_06b0c_row1_col44" class="data row1 col44" >32</td>
                        <td id="T_06b0c_row1_col45" class="data row1 col45" >45</td>
                        <td id="T_06b0c_row1_col46" class="data row1 col46" >46</td>
                        <td id="T_06b0c_row1_col47" class="data row1 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_06b0c_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_06b0c_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_06b0c_row2_col1" class="data row2 col1" >1</td>
                        <td id="T_06b0c_row2_col2" class="data row2 col2" >2</td>
                        <td id="T_06b0c_row2_col3" class="data row2 col3" >3</td>
                        <td id="T_06b0c_row2_col4" class="data row2 col4" >10</td>
                        <td id="T_06b0c_row2_col5" class="data row2 col5" >11</td>
                        <td id="T_06b0c_row2_col6" class="data row2 col6" >12</td>
                        <td id="T_06b0c_row2_col7" class="data row2 col7" >7</td>
                        <td id="T_06b0c_row2_col8" class="data row2 col8" >8</td>
                        <td id="T_06b0c_row2_col9" class="data row2 col9" >9</td>
                        <td id="T_06b0c_row2_col10" class="data row2 col10" >40</td>
                        <td id="T_06b0c_row2_col11" class="data row2 col11" >41</td>
                        <td id="T_06b0c_row2_col12" class="data row2 col12" >42</td>
                        <td id="T_06b0c_row2_col13" class="data row2 col13" >13</td>
                        <td id="T_06b0c_row2_col14" class="data row2 col14" >14</td>
                        <td id="T_06b0c_row2_col15" class="data row2 col15" >15</td>
                        <td id="T_06b0c_row2_col16" class="data row2 col16" >22</td>
                        <td id="T_06b0c_row2_col17" class="data row2 col17" >23</td>
                        <td id="T_06b0c_row2_col18" class="data row2 col18" >16</td>
                        <td id="T_06b0c_row2_col19" class="data row2 col19" >17</td>
                        <td id="T_06b0c_row2_col20" class="data row2 col20" >18</td>
                        <td id="T_06b0c_row2_col21" class="data row2 col21" >19</td>
                        <td id="T_06b0c_row2_col22" class="data row2 col22" >20</td>
                        <td id="T_06b0c_row2_col23" class="data row2 col23" >21</td>
                        <td id="T_06b0c_row2_col24" class="data row2 col24" >6</td>
                        <td id="T_06b0c_row2_col25" class="data row2 col25" >25</td>
                        <td id="T_06b0c_row2_col26" class="data row2 col26" >26</td>
                        <td id="T_06b0c_row2_col27" class="data row2 col27" >27</td>
                        <td id="T_06b0c_row2_col28" class="data row2 col28" >28</td>
                        <td id="T_06b0c_row2_col29" class="data row2 col29" >29</td>
                        <td id="T_06b0c_row2_col30" class="data row2 col30" >4</td>
                        <td id="T_06b0c_row2_col31" class="data row2 col31" >5</td>
                        <td id="T_06b0c_row2_col32" class="data row2 col32" >32</td>
                        <td id="T_06b0c_row2_col33" class="data row2 col33" >33</td>
                        <td id="T_06b0c_row2_col34" class="data row2 col34" >34</td>
                        <td id="T_06b0c_row2_col35" class="data row2 col35" >35</td>
                        <td id="T_06b0c_row2_col36" class="data row2 col36" >36</td>
                        <td id="T_06b0c_row2_col37" class="data row2 col37" >37</td>
                        <td id="T_06b0c_row2_col38" class="data row2 col38" >38</td>
                        <td id="T_06b0c_row2_col39" class="data row2 col39" >39</td>
                        <td id="T_06b0c_row2_col40" class="data row2 col40" >30</td>
                        <td id="T_06b0c_row2_col41" class="data row2 col41" >31</td>
                        <td id="T_06b0c_row2_col42" class="data row2 col42" >24</td>
                        <td id="T_06b0c_row2_col43" class="data row2 col43" >43</td>
                        <td id="T_06b0c_row2_col44" class="data row2 col44" >44</td>
                        <td id="T_06b0c_row2_col45" class="data row2 col45" >45</td>
                        <td id="T_06b0c_row2_col46" class="data row2 col46" >46</td>
                        <td id="T_06b0c_row2_col47" class="data row2 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_06b0c_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_06b0c_row3_col0" class="data row3 col0" >36</td>
                        <td id="T_06b0c_row3_col1" class="data row3 col1" >1</td>
                        <td id="T_06b0c_row3_col2" class="data row3 col2" >2</td>
                        <td id="T_06b0c_row3_col3" class="data row3 col3" >3</td>
                        <td id="T_06b0c_row3_col4" class="data row3 col4" >4</td>
                        <td id="T_06b0c_row3_col5" class="data row3 col5" >5</td>
                        <td id="T_06b0c_row3_col6" class="data row3 col6" >34</td>
                        <td id="T_06b0c_row3_col7" class="data row3 col7" >35</td>
                        <td id="T_06b0c_row3_col8" class="data row3 col8" >14</td>
                        <td id="T_06b0c_row3_col9" class="data row3 col9" >15</td>
                        <td id="T_06b0c_row3_col10" class="data row3 col10" >8</td>
                        <td id="T_06b0c_row3_col11" class="data row3 col11" >9</td>
                        <td id="T_06b0c_row3_col12" class="data row3 col12" >10</td>
                        <td id="T_06b0c_row3_col13" class="data row3 col13" >11</td>
                        <td id="T_06b0c_row3_col14" class="data row3 col14" >12</td>
                        <td id="T_06b0c_row3_col15" class="data row3 col15" >13</td>
                        <td id="T_06b0c_row3_col16" class="data row3 col16" >0</td>
                        <td id="T_06b0c_row3_col17" class="data row3 col17" >17</td>
                        <td id="T_06b0c_row3_col18" class="data row3 col18" >18</td>
                        <td id="T_06b0c_row3_col19" class="data row3 col19" >19</td>
                        <td id="T_06b0c_row3_col20" class="data row3 col20" >20</td>
                        <td id="T_06b0c_row3_col21" class="data row3 col21" >21</td>
                        <td id="T_06b0c_row3_col22" class="data row3 col22" >6</td>
                        <td id="T_06b0c_row3_col23" class="data row3 col23" >7</td>
                        <td id="T_06b0c_row3_col24" class="data row3 col24" >24</td>
                        <td id="T_06b0c_row3_col25" class="data row3 col25" >25</td>
                        <td id="T_06b0c_row3_col26" class="data row3 col26" >26</td>
                        <td id="T_06b0c_row3_col27" class="data row3 col27" >27</td>
                        <td id="T_06b0c_row3_col28" class="data row3 col28" >28</td>
                        <td id="T_06b0c_row3_col29" class="data row3 col29" >29</td>
                        <td id="T_06b0c_row3_col30" class="data row3 col30" >30</td>
                        <td id="T_06b0c_row3_col31" class="data row3 col31" >31</td>
                        <td id="T_06b0c_row3_col32" class="data row3 col32" >32</td>
                        <td id="T_06b0c_row3_col33" class="data row3 col33" >33</td>
                        <td id="T_06b0c_row3_col34" class="data row3 col34" >46</td>
                        <td id="T_06b0c_row3_col35" class="data row3 col35" >47</td>
                        <td id="T_06b0c_row3_col36" class="data row3 col36" >40</td>
                        <td id="T_06b0c_row3_col37" class="data row3 col37" >37</td>
                        <td id="T_06b0c_row3_col38" class="data row3 col38" >38</td>
                        <td id="T_06b0c_row3_col39" class="data row3 col39" >39</td>
                        <td id="T_06b0c_row3_col40" class="data row3 col40" >16</td>
                        <td id="T_06b0c_row3_col41" class="data row3 col41" >41</td>
                        <td id="T_06b0c_row3_col42" class="data row3 col42" >42</td>
                        <td id="T_06b0c_row3_col43" class="data row3 col43" >43</td>
                        <td id="T_06b0c_row3_col44" class="data row3 col44" >44</td>
                        <td id="T_06b0c_row3_col45" class="data row3 col45" >45</td>
                        <td id="T_06b0c_row3_col46" class="data row3 col46" >22</td>
                        <td id="T_06b0c_row3_col47" class="data row3 col47" >23</td>
            </tr>
            <tr>
                        <th id="T_06b0c_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_06b0c_row4_col0" class="data row4 col0" >26</td>
                        <td id="T_06b0c_row4_col1" class="data row4 col1" >27</td>
                        <td id="T_06b0c_row4_col2" class="data row4 col2" >28</td>
                        <td id="T_06b0c_row4_col3" class="data row4 col3" >3</td>
                        <td id="T_06b0c_row4_col4" class="data row4 col4" >4</td>
                        <td id="T_06b0c_row4_col5" class="data row4 col5" >5</td>
                        <td id="T_06b0c_row4_col6" class="data row4 col6" >6</td>
                        <td id="T_06b0c_row4_col7" class="data row4 col7" >7</td>
                        <td id="T_06b0c_row4_col8" class="data row4 col8" >2</td>
                        <td id="T_06b0c_row4_col9" class="data row4 col9" >9</td>
                        <td id="T_06b0c_row4_col10" class="data row4 col10" >10</td>
                        <td id="T_06b0c_row4_col11" class="data row4 col11" >11</td>
                        <td id="T_06b0c_row4_col12" class="data row4 col12" >12</td>
                        <td id="T_06b0c_row4_col13" class="data row4 col13" >13</td>
                        <td id="T_06b0c_row4_col14" class="data row4 col14" >0</td>
                        <td id="T_06b0c_row4_col15" class="data row4 col15" >1</td>
                        <td id="T_06b0c_row4_col16" class="data row4 col16" >16</td>
                        <td id="T_06b0c_row4_col17" class="data row4 col17" >17</td>
                        <td id="T_06b0c_row4_col18" class="data row4 col18" >18</td>
                        <td id="T_06b0c_row4_col19" class="data row4 col19" >19</td>
                        <td id="T_06b0c_row4_col20" class="data row4 col20" >20</td>
                        <td id="T_06b0c_row4_col21" class="data row4 col21" >21</td>
                        <td id="T_06b0c_row4_col22" class="data row4 col22" >22</td>
                        <td id="T_06b0c_row4_col23" class="data row4 col23" >23</td>
                        <td id="T_06b0c_row4_col24" class="data row4 col24" >24</td>
                        <td id="T_06b0c_row4_col25" class="data row4 col25" >25</td>
                        <td id="T_06b0c_row4_col26" class="data row4 col26" >44</td>
                        <td id="T_06b0c_row4_col27" class="data row4 col27" >45</td>
                        <td id="T_06b0c_row4_col28" class="data row4 col28" >46</td>
                        <td id="T_06b0c_row4_col29" class="data row4 col29" >29</td>
                        <td id="T_06b0c_row4_col30" class="data row4 col30" >30</td>
                        <td id="T_06b0c_row4_col31" class="data row4 col31" >31</td>
                        <td id="T_06b0c_row4_col32" class="data row4 col32" >38</td>
                        <td id="T_06b0c_row4_col33" class="data row4 col33" >39</td>
                        <td id="T_06b0c_row4_col34" class="data row4 col34" >32</td>
                        <td id="T_06b0c_row4_col35" class="data row4 col35" >33</td>
                        <td id="T_06b0c_row4_col36" class="data row4 col36" >34</td>
                        <td id="T_06b0c_row4_col37" class="data row4 col37" >35</td>
                        <td id="T_06b0c_row4_col38" class="data row4 col38" >36</td>
                        <td id="T_06b0c_row4_col39" class="data row4 col39" >37</td>
                        <td id="T_06b0c_row4_col40" class="data row4 col40" >40</td>
                        <td id="T_06b0c_row4_col41" class="data row4 col41" >41</td>
                        <td id="T_06b0c_row4_col42" class="data row4 col42" >42</td>
                        <td id="T_06b0c_row4_col43" class="data row4 col43" >43</td>
                        <td id="T_06b0c_row4_col44" class="data row4 col44" >14</td>
                        <td id="T_06b0c_row4_col45" class="data row4 col45" >15</td>
                        <td id="T_06b0c_row4_col46" class="data row4 col46" >8</td>
                        <td id="T_06b0c_row4_col47" class="data row4 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_06b0c_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_06b0c_row5_col0" class="data row5 col0" >0</td>
                        <td id="T_06b0c_row5_col1" class="data row5 col1" >1</td>
                        <td id="T_06b0c_row5_col2" class="data row5 col2" >2</td>
                        <td id="T_06b0c_row5_col3" class="data row5 col3" >3</td>
                        <td id="T_06b0c_row5_col4" class="data row5 col4" >4</td>
                        <td id="T_06b0c_row5_col5" class="data row5 col5" >5</td>
                        <td id="T_06b0c_row5_col6" class="data row5 col6" >6</td>
                        <td id="T_06b0c_row5_col7" class="data row5 col7" >7</td>
                        <td id="T_06b0c_row5_col8" class="data row5 col8" >8</td>
                        <td id="T_06b0c_row5_col9" class="data row5 col9" >9</td>
                        <td id="T_06b0c_row5_col10" class="data row5 col10" >10</td>
                        <td id="T_06b0c_row5_col11" class="data row5 col11" >11</td>
                        <td id="T_06b0c_row5_col12" class="data row5 col12" >36</td>
                        <td id="T_06b0c_row5_col13" class="data row5 col13" >37</td>
                        <td id="T_06b0c_row5_col14" class="data row5 col14" >38</td>
                        <td id="T_06b0c_row5_col15" class="data row5 col15" >15</td>
                        <td id="T_06b0c_row5_col16" class="data row5 col16" >16</td>
                        <td id="T_06b0c_row5_col17" class="data row5 col17" >17</td>
                        <td id="T_06b0c_row5_col18" class="data row5 col18" >18</td>
                        <td id="T_06b0c_row5_col19" class="data row5 col19" >19</td>
                        <td id="T_06b0c_row5_col20" class="data row5 col20" >12</td>
                        <td id="T_06b0c_row5_col21" class="data row5 col21" >13</td>
                        <td id="T_06b0c_row5_col22" class="data row5 col22" >14</td>
                        <td id="T_06b0c_row5_col23" class="data row5 col23" >23</td>
                        <td id="T_06b0c_row5_col24" class="data row5 col24" >24</td>
                        <td id="T_06b0c_row5_col25" class="data row5 col25" >25</td>
                        <td id="T_06b0c_row5_col26" class="data row5 col26" >26</td>
                        <td id="T_06b0c_row5_col27" class="data row5 col27" >27</td>
                        <td id="T_06b0c_row5_col28" class="data row5 col28" >20</td>
                        <td id="T_06b0c_row5_col29" class="data row5 col29" >21</td>
                        <td id="T_06b0c_row5_col30" class="data row5 col30" >22</td>
                        <td id="T_06b0c_row5_col31" class="data row5 col31" >31</td>
                        <td id="T_06b0c_row5_col32" class="data row5 col32" >32</td>
                        <td id="T_06b0c_row5_col33" class="data row5 col33" >33</td>
                        <td id="T_06b0c_row5_col34" class="data row5 col34" >34</td>
                        <td id="T_06b0c_row5_col35" class="data row5 col35" >35</td>
                        <td id="T_06b0c_row5_col36" class="data row5 col36" >28</td>
                        <td id="T_06b0c_row5_col37" class="data row5 col37" >29</td>
                        <td id="T_06b0c_row5_col38" class="data row5 col38" >30</td>
                        <td id="T_06b0c_row5_col39" class="data row5 col39" >39</td>
                        <td id="T_06b0c_row5_col40" class="data row5 col40" >46</td>
                        <td id="T_06b0c_row5_col41" class="data row5 col41" >47</td>
                        <td id="T_06b0c_row5_col42" class="data row5 col42" >40</td>
                        <td id="T_06b0c_row5_col43" class="data row5 col43" >41</td>
                        <td id="T_06b0c_row5_col44" class="data row5 col44" >42</td>
                        <td id="T_06b0c_row5_col45" class="data row5 col45" >43</td>
                        <td id="T_06b0c_row5_col46" class="data row5 col46" >44</td>
                        <td id="T_06b0c_row5_col47" class="data row5 col47" >45</td>
            </tr>
    </tbody></table>



Hopefully you should now have some idea what this puzzle is about.

Essentially, the 48 keys are each assigned one of 6 colors, and every group of 8 consecutive keys represent one side of a Rubiks Cube!

Now if you have a real cube lying around, try assigning the label to the rows yourself; it's a fun little exercise.


```python
# update the index with somewhat random letters. 
MOVES = ["U", "R", "F", "L", "B", "D"]
PERMUTATIONS_DF.index = MOVES
PERMUTATIONS_DF.style.set_caption('Permutations visualized with "arbitrary" colors and new index. 🟪 = move to same block').applymap(color_group).apply(color_not_id_in_group,axis=1)
```




<style  type="text/css" >
#T_5cb61_row0_col0,#T_5cb61_row0_col1,#T_5cb61_row0_col2,#T_5cb61_row0_col3,#T_5cb61_row0_col4,#T_5cb61_row0_col5,#T_5cb61_row0_col6,#T_5cb61_row0_col7{
            background-color:  white;
            color:  magenta;
        }#T_5cb61_row0_col8,#T_5cb61_row0_col9,#T_5cb61_row0_col10,#T_5cb61_row0_col19,#T_5cb61_row0_col20,#T_5cb61_row0_col21,#T_5cb61_row0_col22,#T_5cb61_row0_col23,#T_5cb61_row1_col2,#T_5cb61_row1_col3,#T_5cb61_row1_col4,#T_5cb61_row1_col16,#T_5cb61_row1_col17,#T_5cb61_row1_col21,#T_5cb61_row1_col22,#T_5cb61_row1_col23,#T_5cb61_row3_col17,#T_5cb61_row3_col18,#T_5cb61_row3_col19,#T_5cb61_row3_col20,#T_5cb61_row3_col21,#T_5cb61_row3_col40,#T_5cb61_row3_col46,#T_5cb61_row3_col47,#T_5cb61_row4_col16,#T_5cb61_row4_col17,#T_5cb61_row4_col18,#T_5cb61_row4_col19,#T_5cb61_row4_col20,#T_5cb61_row4_col21,#T_5cb61_row4_col22,#T_5cb61_row4_col23,#T_5cb61_row5_col16,#T_5cb61_row5_col17,#T_5cb61_row5_col18,#T_5cb61_row5_col19,#T_5cb61_row5_col23,#T_5cb61_row5_col28,#T_5cb61_row5_col29,#T_5cb61_row5_col30{
            background-color:  green;
        }#T_5cb61_row0_col11,#T_5cb61_row0_col12,#T_5cb61_row0_col13,#T_5cb61_row0_col14,#T_5cb61_row0_col15,#T_5cb61_row0_col32,#T_5cb61_row0_col33,#T_5cb61_row0_col34,#T_5cb61_row1_col8,#T_5cb61_row1_col9,#T_5cb61_row1_col10,#T_5cb61_row1_col11,#T_5cb61_row1_col12,#T_5cb61_row1_col13,#T_5cb61_row1_col14,#T_5cb61_row1_col15,#T_5cb61_row2_col4,#T_5cb61_row2_col5,#T_5cb61_row2_col6,#T_5cb61_row2_col8,#T_5cb61_row2_col9,#T_5cb61_row2_col13,#T_5cb61_row2_col14,#T_5cb61_row2_col15,#T_5cb61_row4_col9,#T_5cb61_row4_col10,#T_5cb61_row4_col11,#T_5cb61_row4_col12,#T_5cb61_row4_col13,#T_5cb61_row4_col44,#T_5cb61_row4_col45,#T_5cb61_row4_col46,#T_5cb61_row5_col8,#T_5cb61_row5_col9,#T_5cb61_row5_col10,#T_5cb61_row5_col11,#T_5cb61_row5_col15,#T_5cb61_row5_col20,#T_5cb61_row5_col21,#T_5cb61_row5_col22{
            background-color:  orange;
        }#T_5cb61_row0_col16,#T_5cb61_row0_col17,#T_5cb61_row0_col18,#T_5cb61_row0_col27,#T_5cb61_row0_col28,#T_5cb61_row0_col29,#T_5cb61_row0_col30,#T_5cb61_row0_col31,#T_5cb61_row2_col25,#T_5cb61_row2_col26,#T_5cb61_row2_col27,#T_5cb61_row2_col28,#T_5cb61_row2_col29,#T_5cb61_row2_col40,#T_5cb61_row2_col41,#T_5cb61_row2_col42,#T_5cb61_row3_col24,#T_5cb61_row3_col25,#T_5cb61_row3_col26,#T_5cb61_row3_col27,#T_5cb61_row3_col28,#T_5cb61_row3_col29,#T_5cb61_row3_col30,#T_5cb61_row3_col31,#T_5cb61_row4_col0,#T_5cb61_row4_col1,#T_5cb61_row4_col2,#T_5cb61_row4_col24,#T_5cb61_row4_col25,#T_5cb61_row4_col29,#T_5cb61_row4_col30,#T_5cb61_row4_col31,#T_5cb61_row5_col24,#T_5cb61_row5_col25,#T_5cb61_row5_col26,#T_5cb61_row5_col27,#T_5cb61_row5_col31,#T_5cb61_row5_col36,#T_5cb61_row5_col37,#T_5cb61_row5_col38{
            background-color:  red;
        }#T_5cb61_row0_col24,#T_5cb61_row0_col25,#T_5cb61_row0_col26,#T_5cb61_row0_col35,#T_5cb61_row0_col36,#T_5cb61_row0_col37,#T_5cb61_row0_col38,#T_5cb61_row0_col39,#T_5cb61_row1_col33,#T_5cb61_row1_col34,#T_5cb61_row1_col35,#T_5cb61_row1_col36,#T_5cb61_row1_col37,#T_5cb61_row1_col42,#T_5cb61_row1_col43,#T_5cb61_row1_col44,#T_5cb61_row2_col32,#T_5cb61_row2_col33,#T_5cb61_row2_col34,#T_5cb61_row2_col35,#T_5cb61_row2_col36,#T_5cb61_row2_col37,#T_5cb61_row2_col38,#T_5cb61_row2_col39,#T_5cb61_row3_col0,#T_5cb61_row3_col6,#T_5cb61_row3_col7,#T_5cb61_row3_col32,#T_5cb61_row3_col33,#T_5cb61_row3_col37,#T_5cb61_row3_col38,#T_5cb61_row3_col39,#T_5cb61_row5_col12,#T_5cb61_row5_col13,#T_5cb61_row5_col14,#T_5cb61_row5_col32,#T_5cb61_row5_col33,#T_5cb61_row5_col34,#T_5cb61_row5_col35,#T_5cb61_row5_col39{
            background-color:  blue;
        }#T_5cb61_row0_col40,#T_5cb61_row0_col41,#T_5cb61_row0_col42,#T_5cb61_row0_col43,#T_5cb61_row0_col44,#T_5cb61_row0_col45,#T_5cb61_row0_col46,#T_5cb61_row0_col47,#T_5cb61_row1_col18,#T_5cb61_row1_col19,#T_5cb61_row1_col20,#T_5cb61_row1_col40,#T_5cb61_row1_col41,#T_5cb61_row1_col45,#T_5cb61_row1_col46,#T_5cb61_row1_col47,#T_5cb61_row2_col10,#T_5cb61_row2_col11,#T_5cb61_row2_col12,#T_5cb61_row2_col43,#T_5cb61_row2_col44,#T_5cb61_row2_col45,#T_5cb61_row2_col46,#T_5cb61_row2_col47,#T_5cb61_row3_col34,#T_5cb61_row3_col35,#T_5cb61_row3_col36,#T_5cb61_row3_col41,#T_5cb61_row3_col42,#T_5cb61_row3_col43,#T_5cb61_row3_col44,#T_5cb61_row3_col45,#T_5cb61_row4_col26,#T_5cb61_row4_col27,#T_5cb61_row4_col28,#T_5cb61_row4_col40,#T_5cb61_row4_col41,#T_5cb61_row4_col42,#T_5cb61_row4_col43,#T_5cb61_row4_col47{
            background-color:  yellow;
        }#T_5cb61_row1_col0,#T_5cb61_row1_col1,#T_5cb61_row1_col5,#T_5cb61_row1_col6,#T_5cb61_row1_col7,#T_5cb61_row1_col32,#T_5cb61_row1_col38,#T_5cb61_row1_col39,#T_5cb61_row2_col0,#T_5cb61_row2_col1,#T_5cb61_row2_col2,#T_5cb61_row2_col3,#T_5cb61_row2_col7,#T_5cb61_row2_col24,#T_5cb61_row2_col30,#T_5cb61_row2_col31,#T_5cb61_row3_col1,#T_5cb61_row3_col2,#T_5cb61_row3_col3,#T_5cb61_row3_col4,#T_5cb61_row3_col5,#T_5cb61_row3_col16,#T_5cb61_row3_col22,#T_5cb61_row3_col23,#T_5cb61_row4_col3,#T_5cb61_row4_col4,#T_5cb61_row4_col5,#T_5cb61_row4_col6,#T_5cb61_row4_col7,#T_5cb61_row4_col8,#T_5cb61_row4_col14,#T_5cb61_row4_col15,#T_5cb61_row5_col0,#T_5cb61_row5_col1,#T_5cb61_row5_col2,#T_5cb61_row5_col3,#T_5cb61_row5_col4,#T_5cb61_row5_col5,#T_5cb61_row5_col6,#T_5cb61_row5_col7{
            background-color:  white;
        }#T_5cb61_row1_col24,#T_5cb61_row1_col25,#T_5cb61_row1_col26,#T_5cb61_row1_col27,#T_5cb61_row1_col28,#T_5cb61_row1_col29,#T_5cb61_row1_col30,#T_5cb61_row1_col31{
            background-color:  red;
            color:  magenta;
        }#T_5cb61_row2_col16,#T_5cb61_row2_col17,#T_5cb61_row2_col18,#T_5cb61_row2_col19,#T_5cb61_row2_col20,#T_5cb61_row2_col21,#T_5cb61_row2_col22,#T_5cb61_row2_col23{
            background-color:  green;
            color:  magenta;
        }#T_5cb61_row3_col8,#T_5cb61_row3_col9,#T_5cb61_row3_col10,#T_5cb61_row3_col11,#T_5cb61_row3_col12,#T_5cb61_row3_col13,#T_5cb61_row3_col14,#T_5cb61_row3_col15{
            background-color:  orange;
            color:  magenta;
        }#T_5cb61_row4_col32,#T_5cb61_row4_col33,#T_5cb61_row4_col34,#T_5cb61_row4_col35,#T_5cb61_row4_col36,#T_5cb61_row4_col37,#T_5cb61_row4_col38,#T_5cb61_row4_col39{
            background-color:  blue;
            color:  magenta;
        }#T_5cb61_row5_col40,#T_5cb61_row5_col41,#T_5cb61_row5_col42,#T_5cb61_row5_col43,#T_5cb61_row5_col44,#T_5cb61_row5_col45,#T_5cb61_row5_col46,#T_5cb61_row5_col47{
            background-color:  yellow;
            color:  magenta;
        }</style>
<table id="T_5cb61_" ><caption>Permutations visualized with "arbitrary" colors and new index. 🟪 = move to same block</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>        <th class="col_heading level0 col18" >18</th>        <th class="col_heading level0 col19" >19</th>        <th class="col_heading level0 col20" >20</th>        <th class="col_heading level0 col21" >21</th>        <th class="col_heading level0 col22" >22</th>        <th class="col_heading level0 col23" >23</th>        <th class="col_heading level0 col24" >24</th>        <th class="col_heading level0 col25" >25</th>        <th class="col_heading level0 col26" >26</th>        <th class="col_heading level0 col27" >27</th>        <th class="col_heading level0 col28" >28</th>        <th class="col_heading level0 col29" >29</th>        <th class="col_heading level0 col30" >30</th>        <th class="col_heading level0 col31" >31</th>        <th class="col_heading level0 col32" >32</th>        <th class="col_heading level0 col33" >33</th>        <th class="col_heading level0 col34" >34</th>        <th class="col_heading level0 col35" >35</th>        <th class="col_heading level0 col36" >36</th>        <th class="col_heading level0 col37" >37</th>        <th class="col_heading level0 col38" >38</th>        <th class="col_heading level0 col39" >39</th>        <th class="col_heading level0 col40" >40</th>        <th class="col_heading level0 col41" >41</th>        <th class="col_heading level0 col42" >42</th>        <th class="col_heading level0 col43" >43</th>        <th class="col_heading level0 col44" >44</th>        <th class="col_heading level0 col45" >45</th>        <th class="col_heading level0 col46" >46</th>        <th class="col_heading level0 col47" >47</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_5cb61_level0_row0" class="row_heading level0 row0" >U</th>
                        <td id="T_5cb61_row0_col0" class="data row0 col0" >6</td>
                        <td id="T_5cb61_row0_col1" class="data row0 col1" >7</td>
                        <td id="T_5cb61_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_5cb61_row0_col3" class="data row0 col3" >1</td>
                        <td id="T_5cb61_row0_col4" class="data row0 col4" >2</td>
                        <td id="T_5cb61_row0_col5" class="data row0 col5" >3</td>
                        <td id="T_5cb61_row0_col6" class="data row0 col6" >4</td>
                        <td id="T_5cb61_row0_col7" class="data row0 col7" >5</td>
                        <td id="T_5cb61_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_5cb61_row0_col9" class="data row0 col9" >17</td>
                        <td id="T_5cb61_row0_col10" class="data row0 col10" >18</td>
                        <td id="T_5cb61_row0_col11" class="data row0 col11" >11</td>
                        <td id="T_5cb61_row0_col12" class="data row0 col12" >12</td>
                        <td id="T_5cb61_row0_col13" class="data row0 col13" >13</td>
                        <td id="T_5cb61_row0_col14" class="data row0 col14" >14</td>
                        <td id="T_5cb61_row0_col15" class="data row0 col15" >15</td>
                        <td id="T_5cb61_row0_col16" class="data row0 col16" >24</td>
                        <td id="T_5cb61_row0_col17" class="data row0 col17" >25</td>
                        <td id="T_5cb61_row0_col18" class="data row0 col18" >26</td>
                        <td id="T_5cb61_row0_col19" class="data row0 col19" >19</td>
                        <td id="T_5cb61_row0_col20" class="data row0 col20" >20</td>
                        <td id="T_5cb61_row0_col21" class="data row0 col21" >21</td>
                        <td id="T_5cb61_row0_col22" class="data row0 col22" >22</td>
                        <td id="T_5cb61_row0_col23" class="data row0 col23" >23</td>
                        <td id="T_5cb61_row0_col24" class="data row0 col24" >32</td>
                        <td id="T_5cb61_row0_col25" class="data row0 col25" >33</td>
                        <td id="T_5cb61_row0_col26" class="data row0 col26" >34</td>
                        <td id="T_5cb61_row0_col27" class="data row0 col27" >27</td>
                        <td id="T_5cb61_row0_col28" class="data row0 col28" >28</td>
                        <td id="T_5cb61_row0_col29" class="data row0 col29" >29</td>
                        <td id="T_5cb61_row0_col30" class="data row0 col30" >30</td>
                        <td id="T_5cb61_row0_col31" class="data row0 col31" >31</td>
                        <td id="T_5cb61_row0_col32" class="data row0 col32" >8</td>
                        <td id="T_5cb61_row0_col33" class="data row0 col33" >9</td>
                        <td id="T_5cb61_row0_col34" class="data row0 col34" >10</td>
                        <td id="T_5cb61_row0_col35" class="data row0 col35" >35</td>
                        <td id="T_5cb61_row0_col36" class="data row0 col36" >36</td>
                        <td id="T_5cb61_row0_col37" class="data row0 col37" >37</td>
                        <td id="T_5cb61_row0_col38" class="data row0 col38" >38</td>
                        <td id="T_5cb61_row0_col39" class="data row0 col39" >39</td>
                        <td id="T_5cb61_row0_col40" class="data row0 col40" >40</td>
                        <td id="T_5cb61_row0_col41" class="data row0 col41" >41</td>
                        <td id="T_5cb61_row0_col42" class="data row0 col42" >42</td>
                        <td id="T_5cb61_row0_col43" class="data row0 col43" >43</td>
                        <td id="T_5cb61_row0_col44" class="data row0 col44" >44</td>
                        <td id="T_5cb61_row0_col45" class="data row0 col45" >45</td>
                        <td id="T_5cb61_row0_col46" class="data row0 col46" >46</td>
                        <td id="T_5cb61_row0_col47" class="data row0 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_5cb61_level0_row1" class="row_heading level0 row1" >R</th>
                        <td id="T_5cb61_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_5cb61_row1_col1" class="data row1 col1" >1</td>
                        <td id="T_5cb61_row1_col2" class="data row1 col2" >18</td>
                        <td id="T_5cb61_row1_col3" class="data row1 col3" >19</td>
                        <td id="T_5cb61_row1_col4" class="data row1 col4" >20</td>
                        <td id="T_5cb61_row1_col5" class="data row1 col5" >5</td>
                        <td id="T_5cb61_row1_col6" class="data row1 col6" >6</td>
                        <td id="T_5cb61_row1_col7" class="data row1 col7" >7</td>
                        <td id="T_5cb61_row1_col8" class="data row1 col8" >8</td>
                        <td id="T_5cb61_row1_col9" class="data row1 col9" >9</td>
                        <td id="T_5cb61_row1_col10" class="data row1 col10" >10</td>
                        <td id="T_5cb61_row1_col11" class="data row1 col11" >11</td>
                        <td id="T_5cb61_row1_col12" class="data row1 col12" >12</td>
                        <td id="T_5cb61_row1_col13" class="data row1 col13" >13</td>
                        <td id="T_5cb61_row1_col14" class="data row1 col14" >14</td>
                        <td id="T_5cb61_row1_col15" class="data row1 col15" >15</td>
                        <td id="T_5cb61_row1_col16" class="data row1 col16" >16</td>
                        <td id="T_5cb61_row1_col17" class="data row1 col17" >17</td>
                        <td id="T_5cb61_row1_col18" class="data row1 col18" >42</td>
                        <td id="T_5cb61_row1_col19" class="data row1 col19" >43</td>
                        <td id="T_5cb61_row1_col20" class="data row1 col20" >44</td>
                        <td id="T_5cb61_row1_col21" class="data row1 col21" >21</td>
                        <td id="T_5cb61_row1_col22" class="data row1 col22" >22</td>
                        <td id="T_5cb61_row1_col23" class="data row1 col23" >23</td>
                        <td id="T_5cb61_row1_col24" class="data row1 col24" >30</td>
                        <td id="T_5cb61_row1_col25" class="data row1 col25" >31</td>
                        <td id="T_5cb61_row1_col26" class="data row1 col26" >24</td>
                        <td id="T_5cb61_row1_col27" class="data row1 col27" >25</td>
                        <td id="T_5cb61_row1_col28" class="data row1 col28" >26</td>
                        <td id="T_5cb61_row1_col29" class="data row1 col29" >27</td>
                        <td id="T_5cb61_row1_col30" class="data row1 col30" >28</td>
                        <td id="T_5cb61_row1_col31" class="data row1 col31" >29</td>
                        <td id="T_5cb61_row1_col32" class="data row1 col32" >4</td>
                        <td id="T_5cb61_row1_col33" class="data row1 col33" >33</td>
                        <td id="T_5cb61_row1_col34" class="data row1 col34" >34</td>
                        <td id="T_5cb61_row1_col35" class="data row1 col35" >35</td>
                        <td id="T_5cb61_row1_col36" class="data row1 col36" >36</td>
                        <td id="T_5cb61_row1_col37" class="data row1 col37" >37</td>
                        <td id="T_5cb61_row1_col38" class="data row1 col38" >2</td>
                        <td id="T_5cb61_row1_col39" class="data row1 col39" >3</td>
                        <td id="T_5cb61_row1_col40" class="data row1 col40" >40</td>
                        <td id="T_5cb61_row1_col41" class="data row1 col41" >41</td>
                        <td id="T_5cb61_row1_col42" class="data row1 col42" >38</td>
                        <td id="T_5cb61_row1_col43" class="data row1 col43" >39</td>
                        <td id="T_5cb61_row1_col44" class="data row1 col44" >32</td>
                        <td id="T_5cb61_row1_col45" class="data row1 col45" >45</td>
                        <td id="T_5cb61_row1_col46" class="data row1 col46" >46</td>
                        <td id="T_5cb61_row1_col47" class="data row1 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_5cb61_level0_row2" class="row_heading level0 row2" >F</th>
                        <td id="T_5cb61_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_5cb61_row2_col1" class="data row2 col1" >1</td>
                        <td id="T_5cb61_row2_col2" class="data row2 col2" >2</td>
                        <td id="T_5cb61_row2_col3" class="data row2 col3" >3</td>
                        <td id="T_5cb61_row2_col4" class="data row2 col4" >10</td>
                        <td id="T_5cb61_row2_col5" class="data row2 col5" >11</td>
                        <td id="T_5cb61_row2_col6" class="data row2 col6" >12</td>
                        <td id="T_5cb61_row2_col7" class="data row2 col7" >7</td>
                        <td id="T_5cb61_row2_col8" class="data row2 col8" >8</td>
                        <td id="T_5cb61_row2_col9" class="data row2 col9" >9</td>
                        <td id="T_5cb61_row2_col10" class="data row2 col10" >40</td>
                        <td id="T_5cb61_row2_col11" class="data row2 col11" >41</td>
                        <td id="T_5cb61_row2_col12" class="data row2 col12" >42</td>
                        <td id="T_5cb61_row2_col13" class="data row2 col13" >13</td>
                        <td id="T_5cb61_row2_col14" class="data row2 col14" >14</td>
                        <td id="T_5cb61_row2_col15" class="data row2 col15" >15</td>
                        <td id="T_5cb61_row2_col16" class="data row2 col16" >22</td>
                        <td id="T_5cb61_row2_col17" class="data row2 col17" >23</td>
                        <td id="T_5cb61_row2_col18" class="data row2 col18" >16</td>
                        <td id="T_5cb61_row2_col19" class="data row2 col19" >17</td>
                        <td id="T_5cb61_row2_col20" class="data row2 col20" >18</td>
                        <td id="T_5cb61_row2_col21" class="data row2 col21" >19</td>
                        <td id="T_5cb61_row2_col22" class="data row2 col22" >20</td>
                        <td id="T_5cb61_row2_col23" class="data row2 col23" >21</td>
                        <td id="T_5cb61_row2_col24" class="data row2 col24" >6</td>
                        <td id="T_5cb61_row2_col25" class="data row2 col25" >25</td>
                        <td id="T_5cb61_row2_col26" class="data row2 col26" >26</td>
                        <td id="T_5cb61_row2_col27" class="data row2 col27" >27</td>
                        <td id="T_5cb61_row2_col28" class="data row2 col28" >28</td>
                        <td id="T_5cb61_row2_col29" class="data row2 col29" >29</td>
                        <td id="T_5cb61_row2_col30" class="data row2 col30" >4</td>
                        <td id="T_5cb61_row2_col31" class="data row2 col31" >5</td>
                        <td id="T_5cb61_row2_col32" class="data row2 col32" >32</td>
                        <td id="T_5cb61_row2_col33" class="data row2 col33" >33</td>
                        <td id="T_5cb61_row2_col34" class="data row2 col34" >34</td>
                        <td id="T_5cb61_row2_col35" class="data row2 col35" >35</td>
                        <td id="T_5cb61_row2_col36" class="data row2 col36" >36</td>
                        <td id="T_5cb61_row2_col37" class="data row2 col37" >37</td>
                        <td id="T_5cb61_row2_col38" class="data row2 col38" >38</td>
                        <td id="T_5cb61_row2_col39" class="data row2 col39" >39</td>
                        <td id="T_5cb61_row2_col40" class="data row2 col40" >30</td>
                        <td id="T_5cb61_row2_col41" class="data row2 col41" >31</td>
                        <td id="T_5cb61_row2_col42" class="data row2 col42" >24</td>
                        <td id="T_5cb61_row2_col43" class="data row2 col43" >43</td>
                        <td id="T_5cb61_row2_col44" class="data row2 col44" >44</td>
                        <td id="T_5cb61_row2_col45" class="data row2 col45" >45</td>
                        <td id="T_5cb61_row2_col46" class="data row2 col46" >46</td>
                        <td id="T_5cb61_row2_col47" class="data row2 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_5cb61_level0_row3" class="row_heading level0 row3" >L</th>
                        <td id="T_5cb61_row3_col0" class="data row3 col0" >36</td>
                        <td id="T_5cb61_row3_col1" class="data row3 col1" >1</td>
                        <td id="T_5cb61_row3_col2" class="data row3 col2" >2</td>
                        <td id="T_5cb61_row3_col3" class="data row3 col3" >3</td>
                        <td id="T_5cb61_row3_col4" class="data row3 col4" >4</td>
                        <td id="T_5cb61_row3_col5" class="data row3 col5" >5</td>
                        <td id="T_5cb61_row3_col6" class="data row3 col6" >34</td>
                        <td id="T_5cb61_row3_col7" class="data row3 col7" >35</td>
                        <td id="T_5cb61_row3_col8" class="data row3 col8" >14</td>
                        <td id="T_5cb61_row3_col9" class="data row3 col9" >15</td>
                        <td id="T_5cb61_row3_col10" class="data row3 col10" >8</td>
                        <td id="T_5cb61_row3_col11" class="data row3 col11" >9</td>
                        <td id="T_5cb61_row3_col12" class="data row3 col12" >10</td>
                        <td id="T_5cb61_row3_col13" class="data row3 col13" >11</td>
                        <td id="T_5cb61_row3_col14" class="data row3 col14" >12</td>
                        <td id="T_5cb61_row3_col15" class="data row3 col15" >13</td>
                        <td id="T_5cb61_row3_col16" class="data row3 col16" >0</td>
                        <td id="T_5cb61_row3_col17" class="data row3 col17" >17</td>
                        <td id="T_5cb61_row3_col18" class="data row3 col18" >18</td>
                        <td id="T_5cb61_row3_col19" class="data row3 col19" >19</td>
                        <td id="T_5cb61_row3_col20" class="data row3 col20" >20</td>
                        <td id="T_5cb61_row3_col21" class="data row3 col21" >21</td>
                        <td id="T_5cb61_row3_col22" class="data row3 col22" >6</td>
                        <td id="T_5cb61_row3_col23" class="data row3 col23" >7</td>
                        <td id="T_5cb61_row3_col24" class="data row3 col24" >24</td>
                        <td id="T_5cb61_row3_col25" class="data row3 col25" >25</td>
                        <td id="T_5cb61_row3_col26" class="data row3 col26" >26</td>
                        <td id="T_5cb61_row3_col27" class="data row3 col27" >27</td>
                        <td id="T_5cb61_row3_col28" class="data row3 col28" >28</td>
                        <td id="T_5cb61_row3_col29" class="data row3 col29" >29</td>
                        <td id="T_5cb61_row3_col30" class="data row3 col30" >30</td>
                        <td id="T_5cb61_row3_col31" class="data row3 col31" >31</td>
                        <td id="T_5cb61_row3_col32" class="data row3 col32" >32</td>
                        <td id="T_5cb61_row3_col33" class="data row3 col33" >33</td>
                        <td id="T_5cb61_row3_col34" class="data row3 col34" >46</td>
                        <td id="T_5cb61_row3_col35" class="data row3 col35" >47</td>
                        <td id="T_5cb61_row3_col36" class="data row3 col36" >40</td>
                        <td id="T_5cb61_row3_col37" class="data row3 col37" >37</td>
                        <td id="T_5cb61_row3_col38" class="data row3 col38" >38</td>
                        <td id="T_5cb61_row3_col39" class="data row3 col39" >39</td>
                        <td id="T_5cb61_row3_col40" class="data row3 col40" >16</td>
                        <td id="T_5cb61_row3_col41" class="data row3 col41" >41</td>
                        <td id="T_5cb61_row3_col42" class="data row3 col42" >42</td>
                        <td id="T_5cb61_row3_col43" class="data row3 col43" >43</td>
                        <td id="T_5cb61_row3_col44" class="data row3 col44" >44</td>
                        <td id="T_5cb61_row3_col45" class="data row3 col45" >45</td>
                        <td id="T_5cb61_row3_col46" class="data row3 col46" >22</td>
                        <td id="T_5cb61_row3_col47" class="data row3 col47" >23</td>
            </tr>
            <tr>
                        <th id="T_5cb61_level0_row4" class="row_heading level0 row4" >B</th>
                        <td id="T_5cb61_row4_col0" class="data row4 col0" >26</td>
                        <td id="T_5cb61_row4_col1" class="data row4 col1" >27</td>
                        <td id="T_5cb61_row4_col2" class="data row4 col2" >28</td>
                        <td id="T_5cb61_row4_col3" class="data row4 col3" >3</td>
                        <td id="T_5cb61_row4_col4" class="data row4 col4" >4</td>
                        <td id="T_5cb61_row4_col5" class="data row4 col5" >5</td>
                        <td id="T_5cb61_row4_col6" class="data row4 col6" >6</td>
                        <td id="T_5cb61_row4_col7" class="data row4 col7" >7</td>
                        <td id="T_5cb61_row4_col8" class="data row4 col8" >2</td>
                        <td id="T_5cb61_row4_col9" class="data row4 col9" >9</td>
                        <td id="T_5cb61_row4_col10" class="data row4 col10" >10</td>
                        <td id="T_5cb61_row4_col11" class="data row4 col11" >11</td>
                        <td id="T_5cb61_row4_col12" class="data row4 col12" >12</td>
                        <td id="T_5cb61_row4_col13" class="data row4 col13" >13</td>
                        <td id="T_5cb61_row4_col14" class="data row4 col14" >0</td>
                        <td id="T_5cb61_row4_col15" class="data row4 col15" >1</td>
                        <td id="T_5cb61_row4_col16" class="data row4 col16" >16</td>
                        <td id="T_5cb61_row4_col17" class="data row4 col17" >17</td>
                        <td id="T_5cb61_row4_col18" class="data row4 col18" >18</td>
                        <td id="T_5cb61_row4_col19" class="data row4 col19" >19</td>
                        <td id="T_5cb61_row4_col20" class="data row4 col20" >20</td>
                        <td id="T_5cb61_row4_col21" class="data row4 col21" >21</td>
                        <td id="T_5cb61_row4_col22" class="data row4 col22" >22</td>
                        <td id="T_5cb61_row4_col23" class="data row4 col23" >23</td>
                        <td id="T_5cb61_row4_col24" class="data row4 col24" >24</td>
                        <td id="T_5cb61_row4_col25" class="data row4 col25" >25</td>
                        <td id="T_5cb61_row4_col26" class="data row4 col26" >44</td>
                        <td id="T_5cb61_row4_col27" class="data row4 col27" >45</td>
                        <td id="T_5cb61_row4_col28" class="data row4 col28" >46</td>
                        <td id="T_5cb61_row4_col29" class="data row4 col29" >29</td>
                        <td id="T_5cb61_row4_col30" class="data row4 col30" >30</td>
                        <td id="T_5cb61_row4_col31" class="data row4 col31" >31</td>
                        <td id="T_5cb61_row4_col32" class="data row4 col32" >38</td>
                        <td id="T_5cb61_row4_col33" class="data row4 col33" >39</td>
                        <td id="T_5cb61_row4_col34" class="data row4 col34" >32</td>
                        <td id="T_5cb61_row4_col35" class="data row4 col35" >33</td>
                        <td id="T_5cb61_row4_col36" class="data row4 col36" >34</td>
                        <td id="T_5cb61_row4_col37" class="data row4 col37" >35</td>
                        <td id="T_5cb61_row4_col38" class="data row4 col38" >36</td>
                        <td id="T_5cb61_row4_col39" class="data row4 col39" >37</td>
                        <td id="T_5cb61_row4_col40" class="data row4 col40" >40</td>
                        <td id="T_5cb61_row4_col41" class="data row4 col41" >41</td>
                        <td id="T_5cb61_row4_col42" class="data row4 col42" >42</td>
                        <td id="T_5cb61_row4_col43" class="data row4 col43" >43</td>
                        <td id="T_5cb61_row4_col44" class="data row4 col44" >14</td>
                        <td id="T_5cb61_row4_col45" class="data row4 col45" >15</td>
                        <td id="T_5cb61_row4_col46" class="data row4 col46" >8</td>
                        <td id="T_5cb61_row4_col47" class="data row4 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_5cb61_level0_row5" class="row_heading level0 row5" >D</th>
                        <td id="T_5cb61_row5_col0" class="data row5 col0" >0</td>
                        <td id="T_5cb61_row5_col1" class="data row5 col1" >1</td>
                        <td id="T_5cb61_row5_col2" class="data row5 col2" >2</td>
                        <td id="T_5cb61_row5_col3" class="data row5 col3" >3</td>
                        <td id="T_5cb61_row5_col4" class="data row5 col4" >4</td>
                        <td id="T_5cb61_row5_col5" class="data row5 col5" >5</td>
                        <td id="T_5cb61_row5_col6" class="data row5 col6" >6</td>
                        <td id="T_5cb61_row5_col7" class="data row5 col7" >7</td>
                        <td id="T_5cb61_row5_col8" class="data row5 col8" >8</td>
                        <td id="T_5cb61_row5_col9" class="data row5 col9" >9</td>
                        <td id="T_5cb61_row5_col10" class="data row5 col10" >10</td>
                        <td id="T_5cb61_row5_col11" class="data row5 col11" >11</td>
                        <td id="T_5cb61_row5_col12" class="data row5 col12" >36</td>
                        <td id="T_5cb61_row5_col13" class="data row5 col13" >37</td>
                        <td id="T_5cb61_row5_col14" class="data row5 col14" >38</td>
                        <td id="T_5cb61_row5_col15" class="data row5 col15" >15</td>
                        <td id="T_5cb61_row5_col16" class="data row5 col16" >16</td>
                        <td id="T_5cb61_row5_col17" class="data row5 col17" >17</td>
                        <td id="T_5cb61_row5_col18" class="data row5 col18" >18</td>
                        <td id="T_5cb61_row5_col19" class="data row5 col19" >19</td>
                        <td id="T_5cb61_row5_col20" class="data row5 col20" >12</td>
                        <td id="T_5cb61_row5_col21" class="data row5 col21" >13</td>
                        <td id="T_5cb61_row5_col22" class="data row5 col22" >14</td>
                        <td id="T_5cb61_row5_col23" class="data row5 col23" >23</td>
                        <td id="T_5cb61_row5_col24" class="data row5 col24" >24</td>
                        <td id="T_5cb61_row5_col25" class="data row5 col25" >25</td>
                        <td id="T_5cb61_row5_col26" class="data row5 col26" >26</td>
                        <td id="T_5cb61_row5_col27" class="data row5 col27" >27</td>
                        <td id="T_5cb61_row5_col28" class="data row5 col28" >20</td>
                        <td id="T_5cb61_row5_col29" class="data row5 col29" >21</td>
                        <td id="T_5cb61_row5_col30" class="data row5 col30" >22</td>
                        <td id="T_5cb61_row5_col31" class="data row5 col31" >31</td>
                        <td id="T_5cb61_row5_col32" class="data row5 col32" >32</td>
                        <td id="T_5cb61_row5_col33" class="data row5 col33" >33</td>
                        <td id="T_5cb61_row5_col34" class="data row5 col34" >34</td>
                        <td id="T_5cb61_row5_col35" class="data row5 col35" >35</td>
                        <td id="T_5cb61_row5_col36" class="data row5 col36" >28</td>
                        <td id="T_5cb61_row5_col37" class="data row5 col37" >29</td>
                        <td id="T_5cb61_row5_col38" class="data row5 col38" >30</td>
                        <td id="T_5cb61_row5_col39" class="data row5 col39" >39</td>
                        <td id="T_5cb61_row5_col40" class="data row5 col40" >46</td>
                        <td id="T_5cb61_row5_col41" class="data row5 col41" >47</td>
                        <td id="T_5cb61_row5_col42" class="data row5 col42" >40</td>
                        <td id="T_5cb61_row5_col43" class="data row5 col43" >41</td>
                        <td id="T_5cb61_row5_col44" class="data row5 col44" >42</td>
                        <td id="T_5cb61_row5_col45" class="data row5 col45" >43</td>
                        <td id="T_5cb61_row5_col46" class="data row5 col46" >44</td>
                        <td id="T_5cb61_row5_col47" class="data row5 col47" >45</td>
            </tr>
    </tbody></table>



We now know what each row does to the cube, so the `sol` array we provide to the program must be the list of moves which solves the cube.

We won't write our own solver, since many great ones already exist.
The one we use is [rubiks-cube-solver.com](https://rubiks-cube-solver.com/solution.php?cube=0611114524521325561451432266332641332352452334554461666) which has a nice interface for inputting the faces (the link should take you to a page where the cube is already input).

Let's take our `initial_value` array and display it in a more readable way.


```python
def color_initial_values(val):
    return f'background-color: {COLORS[val]}' 
    
CUBE_MAP = {
    0: (0,0),
    1: (0,1),
    2: (0,2),
    3: (1,2),
    4: (2,2),
    5: (2,1),
    6: (2,0),
    7: (1,0),
}

FACES = np.zeros((6,3,3), dtype=np.int8)

for i in range(6):
    FACES[i, 1, 1] = i

for i, v in enumerate(INITIAL_VALUES):
    color = v
    pos = i % 8
    c = i //8
    x, y = CUBE_MAP[pos]
    FACES[c,x, y] = color
# display(CUBE_DF.style.applymap(color_initial_values))

for i in range(6):
    face = FACES[i, :, :]
    display(pd.DataFrame(face).style.applymap(color_initial_values))
```


<style  type="text/css" >
#T_e25fe_row0_col0{
            background-color:  yellow;
        }#T_e25fe_row0_col1,#T_e25fe_row0_col2,#T_e25fe_row1_col0,#T_e25fe_row1_col1{
            background-color:  white;
        }#T_e25fe_row1_col2,#T_e25fe_row2_col2{
            background-color:  red;
        }#T_e25fe_row2_col0{
            background-color:  blue;
        }#T_e25fe_row2_col1{
            background-color:  orange;
        }</style>
<table id="T_e25fe_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_e25fe_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_e25fe_row0_col0" class="data row0 col0" >5</td>
                        <td id="T_e25fe_row0_col1" class="data row0 col1" >0</td>
                        <td id="T_e25fe_row0_col2" class="data row0 col2" >0</td>
            </tr>
            <tr>
                        <th id="T_e25fe_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_e25fe_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_e25fe_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_e25fe_row1_col2" class="data row1 col2" >3</td>
            </tr>
            <tr>
                        <th id="T_e25fe_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_e25fe_row2_col0" class="data row2 col0" >4</td>
                        <td id="T_e25fe_row2_col1" class="data row2 col1" >1</td>
                        <td id="T_e25fe_row2_col2" class="data row2 col2" >3</td>
            </tr>
    </tbody></table>



<style  type="text/css" >
#T_ee531_row0_col0,#T_ee531_row1_col2,#T_ee531_row2_col0{
            background-color:  blue;
        }#T_ee531_row0_col1,#T_ee531_row1_col1{
            background-color:  orange;
        }#T_ee531_row0_col2,#T_ee531_row2_col2{
            background-color:  white;
        }#T_ee531_row1_col0{
            background-color:  green;
        }#T_ee531_row2_col1{
            background-color:  yellow;
        }</style>
<table id="T_ee531_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ee531_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_ee531_row0_col0" class="data row0 col0" >4</td>
                        <td id="T_ee531_row0_col1" class="data row0 col1" >1</td>
                        <td id="T_ee531_row0_col2" class="data row0 col2" >0</td>
            </tr>
            <tr>
                        <th id="T_ee531_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_ee531_row1_col0" class="data row1 col0" >2</td>
                        <td id="T_ee531_row1_col1" class="data row1 col1" >1</td>
                        <td id="T_ee531_row1_col2" class="data row1 col2" >4</td>
            </tr>
            <tr>
                        <th id="T_ee531_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_ee531_row2_col0" class="data row2 col0" >4</td>
                        <td id="T_ee531_row2_col1" class="data row2 col1" >5</td>
                        <td id="T_ee531_row2_col2" class="data row2 col2" >0</td>
            </tr>
    </tbody></table>



<style  type="text/css" >
#T_47616_row0_col0,#T_47616_row1_col0{
            background-color:  red;
        }#T_47616_row0_col1{
            background-color:  blue;
        }#T_47616_row0_col2{
            background-color:  white;
        }#T_47616_row1_col1{
            background-color:  green;
        }#T_47616_row1_col2,#T_47616_row2_col0{
            background-color:  orange;
        }#T_47616_row2_col1,#T_47616_row2_col2{
            background-color:  yellow;
        }</style>
<table id="T_47616_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_47616_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_47616_row0_col0" class="data row0 col0" >3</td>
                        <td id="T_47616_row0_col1" class="data row0 col1" >4</td>
                        <td id="T_47616_row0_col2" class="data row0 col2" >0</td>
            </tr>
            <tr>
                        <th id="T_47616_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_47616_row1_col0" class="data row1 col0" >3</td>
                        <td id="T_47616_row1_col1" class="data row1 col1" >2</td>
                        <td id="T_47616_row1_col2" class="data row1 col2" >1</td>
            </tr>
            <tr>
                        <th id="T_47616_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_47616_row2_col0" class="data row2 col0" >1</td>
                        <td id="T_47616_row2_col1" class="data row2 col1" >5</td>
                        <td id="T_47616_row2_col2" class="data row2 col2" >5</td>
            </tr>
    </tbody></table>



<style  type="text/css" >
#T_47203_row0_col0,#T_47203_row0_col1,#T_47203_row2_col0,#T_47203_row2_col1{
            background-color:  green;
        }#T_47203_row0_col2,#T_47203_row2_col2{
            background-color:  orange;
        }#T_47203_row1_col0{
            background-color:  yellow;
        }#T_47203_row1_col1{
            background-color:  red;
        }#T_47203_row1_col2{
            background-color:  white;
        }</style>
<table id="T_47203_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_47203_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_47203_row0_col0" class="data row0 col0" >2</td>
                        <td id="T_47203_row0_col1" class="data row0 col1" >2</td>
                        <td id="T_47203_row0_col2" class="data row0 col2" >1</td>
            </tr>
            <tr>
                        <th id="T_47203_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_47203_row1_col0" class="data row1 col0" >5</td>
                        <td id="T_47203_row1_col1" class="data row1 col1" >3</td>
                        <td id="T_47203_row1_col2" class="data row1 col2" >0</td>
            </tr>
            <tr>
                        <th id="T_47203_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_47203_row2_col0" class="data row2 col0" >2</td>
                        <td id="T_47203_row2_col1" class="data row2 col1" >2</td>
                        <td id="T_47203_row2_col2" class="data row2 col2" >1</td>
            </tr>
    </tbody></table>



<style  type="text/css" >
#T_88455_row0_col0,#T_88455_row2_col0,#T_88455_row2_col1{
            background-color:  green;
        }#T_88455_row0_col1,#T_88455_row1_col1{
            background-color:  blue;
        }#T_88455_row0_col2,#T_88455_row1_col2{
            background-color:  orange;
        }#T_88455_row1_col0,#T_88455_row2_col2{
            background-color:  red;
        }</style>
<table id="T_88455_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_88455_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_88455_row0_col0" class="data row0 col0" >2</td>
                        <td id="T_88455_row0_col1" class="data row0 col1" >4</td>
                        <td id="T_88455_row0_col2" class="data row0 col2" >1</td>
            </tr>
            <tr>
                        <th id="T_88455_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_88455_row1_col0" class="data row1 col0" >3</td>
                        <td id="T_88455_row1_col1" class="data row1 col1" >4</td>
                        <td id="T_88455_row1_col2" class="data row1 col2" >1</td>
            </tr>
            <tr>
                        <th id="T_88455_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_88455_row2_col0" class="data row2 col0" >2</td>
                        <td id="T_88455_row2_col1" class="data row2 col1" >2</td>
                        <td id="T_88455_row2_col2" class="data row2 col2" >3</td>
            </tr>
    </tbody></table>



<style  type="text/css" >
#T_e0643_row0_col0,#T_e0643_row0_col1{
            background-color:  blue;
        }#T_e0643_row0_col2,#T_e0643_row1_col0{
            background-color:  red;
        }#T_e0643_row1_col1,#T_e0643_row2_col0,#T_e0643_row2_col1,#T_e0643_row2_col2{
            background-color:  yellow;
        }#T_e0643_row1_col2{
            background-color:  white;
        }</style>
<table id="T_e0643_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_e0643_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_e0643_row0_col0" class="data row0 col0" >4</td>
                        <td id="T_e0643_row0_col1" class="data row0 col1" >4</td>
                        <td id="T_e0643_row0_col2" class="data row0 col2" >3</td>
            </tr>
            <tr>
                        <th id="T_e0643_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_e0643_row1_col0" class="data row1 col0" >3</td>
                        <td id="T_e0643_row1_col1" class="data row1 col1" >5</td>
                        <td id="T_e0643_row1_col2" class="data row1 col2" >0</td>
            </tr>
            <tr>
                        <th id="T_e0643_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_e0643_row2_col0" class="data row2 col0" >5</td>
                        <td id="T_e0643_row2_col1" class="data row2 col1" >5</td>
                        <td id="T_e0643_row2_col2" class="data row2 col2" >5</td>
            </tr>
    </tbody></table>


Notice that the moves are only the clockwise ones, but the solution we obtain online assumes we can move each face in either direction. 


```python
CUBE_SOL_SHORT = [
    "L", "U", "L'", "U2", "F'", "D'", "F2", "B'", "U'", "L'", "F2", "U'", "D'", "L2",
    "U", "B2", "U", "B2", "R2", "L2"
    ]
```

This little snippet should convert any *2*s and *'*s to the appropriate number of repetitions of the move.

We also map the move letter to the original matrix row.


```python
MOVES_MAP = { v:i for i, v in  enumerate(MOVES)}

def convert_moves(m: str) -> list[str]:
    if len(m) == 2:
        if m[1] == "'":
            return 3*[m[0]]
        if m[1] == "2":
            return 2*[m[0]]
    else:
        return [m]
SOL = list(map(lambda x: MOVES_MAP[x],(itertools.chain(*map(convert_moves, CUBE_SOL_SHORT)))))
print("sol:",SOL)
```

    sol: [3, 0, 3, 3, 3, 0, 0, 2, 2, 2, 5, 5, 5, 2, 2, 4, 4, 4, 0, 0, 0, 3, 3, 3, 2, 2, 0, 0, 0, 5, 5, 5, 3, 3, 0, 4, 4, 0, 4, 4, 1, 1, 3, 3]


Finally, we can check that our solution works as intended with our Python translations of the original Cairo functions.


```python
state = { i:v for i,v in enumerate(INITIAL_VALUES)}

run(state, DATA, SOL)

solved_state = [x for _, x in state.items()]
assert solved_state == list(FINAL_STATE_VALUES), "state is not solved"
```

## Writing the hints

The last, final step is to write the appropriate solutions we found along the way as Cairo hints. 

### `get_initial_value()`

The hint we provide here is simply the `initial_value` list we found previously. 
We need to write the 49 elements (including the initial 48) in the `initial_value` array.

```
func get_initial_value{hash_ptr : HashBuiltin*}() -> (initial_value : felt*):
    alloc_locals
    let (local initial_value : felt*) = alloc()
    %{
    initial_value = [48,
        5, 0, 0, 3, 3, 1, 4, 0,
        4, 1, 0, 4, 0, 5, 4, 2,
        3, 4, 0, 1, 5, 5, 1, 3,
        2, 2, 1, 0, 1, 2, 2, 5,
        2, 4, 1, 1, 3, 2, 2, 3,
        4, 4, 3, 0, 5, 5, 5, 3
    ]
    segments.write_arg(ids.initial_value, initial_value)
    %}
    # ...
    return (initial_value=initial_value + 1)
end

```

### `main()`

The `run` function expects an array `sol` and its length `sol_size`, which are easy to set using `segements.write_arg`.

We also need to initialize the `state` with all the state updates we performed along the way.
This took some time to figure out, but in the end I resorted to re-running the `run` function and saving the DictAccesses.

This is a bit tedious, and there are most likely better ways to do this, but it works!


```python

func main{output_ptr : felt*, pedersen_ptr : HashBuiltin*, range_check_ptr}():
    alloc_locals
    let (local data : felt*) = get_data()
    let (local sol : felt*) = alloc()
    local sol_size

    let (state : DictAccess*) = alloc()
    local state_start : DictAccess* = state
    %{
    FINAL_SOL = [
        3, 0, 3, 3, 3, 0, 0, 2,
        2, 2, 5, 5, 5, 2, 2, 4,
        4, 4, 0, 0, 0, 3, 3, 3,
        2, 2, 0, 0, 0, 5, 5, 5,
        3, 3, 0, 4, 4, 0, 4, 4,
        1, 1, 3, 3]

    segments.write_arg(ids.sol, FINAL_SOL)
    ids.sol_size = len(FINAL_SOL)

    IV = [
        5, 0, 0, 3, 3, 1, 4, 0,
        4, 1, 0, 4, 0, 5, 4, 2,
        3, 4, 0, 1, 5, 5, 1, 3,
        2, 2, 1, 0, 1, 2, 2, 5,
        2, 4, 1, 1, 3, 2, 2, 3,
        4, 4, 3, 0, 5, 5, 5, 3
    ]
    DATA = [
        0, 2, 4, 6, 1, 3, 5, 7, 8, 32, 24, 16, 9, 33, 25, 17, 10, 34, 26, 18, 24, 26, 28, 30, 25, 27, 29, 31, 4, 32, 44, 
        20, 3, 39, 43, 19, 2, 38, 42, 18, 16, 18, 20, 22, 17, 19, 21, 23, 6, 24, 42, 12, 5, 31, 41, 11, 4, 30, 40, 10,
        8, 10, 12, 14, 9, 11, 13, 15, 0, 16, 40, 36, 7, 23, 47, 35, 6, 22, 46, 34, 32, 34, 36, 38, 33, 35, 37, 39, 2, 8,
        46, 28, 1, 15, 45, 27, 0, 14, 44, 26, 40, 42, 44, 46, 41, 43, 45, 47, 22, 30, 38, 14, 21, 29, 37, 13, 20, 28, 36, 12
    ]

    # run iterates over all elements in the solution, and 
    def run(sol):
        state = IV.copy()
        da = []
        for s in sol:
            for i in range(5):
                key0, key1, key2, key3 = DATA[20*s + 4*i:20*s + 4*i + 4]
                da += [key0, state[key0], state[key3]]
                da += [key1, state[key1], state[key0]]
                da += [key2, state[key2], state[key1]]
                da += [key3, state[key3], state[key2]]
                state[key0], state[key1], state[key2], state[key3] = state[key3], state[key0], state[key1], state[key2]
        return da

    dict_access = run(FINAL_SOL)
    segments.write_arg(ids.state.address_, dict_access)

    memory[ids.output_ptr] = 0x0BADBEEF
    %}

    # ...

    let output_ptr = output_ptr + 1
    return ()
end


```
