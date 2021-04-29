---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Amun: Solving the last Cairo-Games challenge"
subtitle: ""
summary: ""
authors: [Adrian Hamelink]
tags: ["Cairo", "Python"]
categories: []
date: 2021-04-29T21:26:00+02:00
lastmod: 2021-04-29T21:26:00+02:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
# yle>
# <tabl
---

<!-- # Amun: Solving the last Cairo-Games challenge -->

For their second edition of the [Cairo Games](https://www.cairo-lang.org/the-cairo-games/puzzles2/), StarkWare have proposed 5 challenges for testing your Cairo skills.

Cairo is a specialized language used for _writing provable programs_.

Its syntax is a mix of many languages (it is built on top of Python), but it has some strong limitations that we will need to work with.
For me the two facts really set it apart:

1. All objects in memory must be elements in the finite field $ \mathbb{F}\_p $ where $p = 2^{251} + 17 \cdot 2^{192} + 1$.
2. Memory is immutable, which means that Cairo borrows a lot of ideas from functional languages (using recursion instead of loops for example).

Cairo itself has many uses beyond simply solving challenges (like building an [AMM](https://www.cairo-lang.org/docs/hello_cairo/amm.html) or [voting system](https://www.cairo-lang.org/docs/hello_cairo/voting.html)).
In these challenges, we are given a correct (but not yet compilable) Cairo program which asserts the validitity of some input.
Our job will be to figure out what the program actually verifies, and build an apporopriate solution, all without modifying the original program.

We will write our solution inside of _hints_ which are tiny snippets of code we embed in the source code.
When the program gets built, these hints tell the compiler what values it should expect in certain memory locations.

This notebook is the write-up of my solution, which mirror my own thought process that led me to the solution.

We will be proceeding in 3 parts:

1. Understanding what the program verifies
2. Solving the problem
3. Building the hints

If you want to try this code yourself, you will need to install the Cairo toolchain, as well as basic Python libraries (numpy, pandas, ...).

## 1. Exploratory Challenge Analysis

In this section, we'll try to understand what all the functions do, and what type of solution it is expecting.

### 1.1 `main()` (1/2)

The Cairo program starts with the `main()` function, and we'll focus for now on the first half.

```js
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

The `main()` function seems to be doing a couple of things:

1. Populate an array `data` with `get_data()`
2. Expect a solution array `sol` of size `sol_size`.
3. Process the solution using `data`, and storing the result in `state`. Here, `state` is represented as a [DictAccess](https://www.cairo-lang.org/docs/hello_cairo/dict.html?) list.
   We can think of it as a list of all updates to a dictionary, but we'll come back to this later.

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
Since sol is an array of `felt` (field elements), we'll need to write a hint like this just after the declaration of `sol` and `sol_size`:

```python
%{
    # our solution goes here
    SOL = [???]
    segments.write_arg(ids.sol, SOL)
    ids.sol_size = len(SOL)
%}
```

If we input some dummy solution in `SOL`, we see that our previous error disapears, and is replaced with another one later in the execution.

### 1.2 `get_data()`

At the very start of the program's execution, the `data` location gets populated by `get_data()`:

```js
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
Instead, we are actually writing a new value in the next memory slot, so that we end up with one contiguous array of values.
The function then returns the adress of `a0` which points to the start of the array.

To better understand how this works in more detail, I recommend reading [How Cairo Works](https://www.cairo-lang.org/docs/how_cairo_works/index.html).

We'll need to study this data later, so for now we'll store these values in the `DATA` variable and run some basic analysis.

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

### 1.3 `run()` and `cycle()`

```js
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

**Each element in `sol` must be in the range $\{0, ..., 5\}$.**

Next, each loop of `run()` calls `cycle()` 5 times with varying inputs.
We see that `cycle()` only reads 4 elements of the array it is given, which means we are feeding it 5 blocks of 4 values, or 20 values in total.
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
Concretely, we can view it as a list of tuples `(key, prev_value, new_value)`.
This corresponds with `DictAccess` struct defined in [starkware/cairo/common/dict_access.cairo](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/cairo/common/dict_access.cairo).

First, we create 4 local references to the next `DictAccess` elements, which represent the dictionary updates we are about to perform.
The keys that will be updated are defined by the 4 values in `values`, so we set the `key`s of each `DictAccess` accordingly.
A cyclic permutation is then applied to the items referenced by the 4 keys.

To make this whole step easier to understand, we rewrite it in Python, and verify that it runs in a similar way.
The main difference with the Cairo version is that we use a simple read-write dictionary to store the state.

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
        assert 0 <= s and s <= 5, "sol contains an element outside [0,5]"
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

1. `state` is a dictionary that contains some initial data referenced by the keys $\{0, ..., 48\}$.
2. `data` can be seen as a 6x20 matrix where each row contains a subset of $\{0, ..., 48\}$.
3. when iterating over `sol`, each element references row of `data` which itself references the keys that will be permuted.
4. since all rows of `data` are duplicate-free, we can view them as permutations over $\{0, ..., 48\}$.
5. each permutation is a composition of 5 cyclic permutations of order 4 each, so all permutation have order 4 (by _order_ we mean that applying the permutation 4 times is the same as doing nothing).

### 1.4 `main()` (2/2)

After generating the appropriate `state` from the solution, we are interested in the second half of `main()` which is responsible for verifying that our provided solution was correct.
Hopefully, understanding what this code does will give us more hints about how we should construct our solution.

```javascript
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

1. The `state` that we have produced isn't particularly useful in its current form, since it represents all updates we performed along the way.
   Applying `squash_dict()` to `state` create a list of tuples `(key, initial_value, final_value)` sorted by `key`.
   The `assert` statement also tells us that the the size of the `squashed_dict` is 48, which means that our solution will affect all keys.

2. `get_initial_value()` most likely provides an array of elements which correspond to the initial `state`.

3. `verify()` should check wether our processed solution is coherent with `initial_value`.

So far, we didn't learn a lot more, so let's dig deeper into the functions.

### 1.5 `get_initial_value()`

```javascript
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

First, `let (local initial_value : felt*) = alloc()` indicates that we will most likely have to provide hints to the compiler about the actual values in the `initial_value` array, since it is not given any initial value.

The first assertion: `assert [initial_value] = 48` nicely tells us that `initial_value[0] == 48`, which is going to help later.

Looking at the return statement however, we see that this value is not actually returned as part of `initial_value`.

We'll now try to understand the statements of the form:

```
    let (res) = hash_chain(initial_value + idx)
    assert res = ????????
```

The `hash_chain(data_ptr : felt*)` function is defined in [starkware/cairo/common/hash_chain.cairo](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/cairo/common/hash_chain.cairo), which provides the following comment:

> Computes a hash chain of a sequence whose length is given at [data_ptr] and the data starts at
> data_ptr + 1. The hash is calculated backwards (from the highest memory address to the lowest).
>
> For example, for the 3-element sequence [x, y, z] the hash is:
>
> h(3, h(x, h(y, z)))
>
> If data_length = 0, the function does not return (takes more than field prime steps).

The actual hash function used is the "Starkware Pedersen" one defined in [starkware/crypto/signature/fast_pedersen_hash.py](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/crypto/starkware/crypto/signature/fast_pedersen_hash.py) (it was mentioned in the hints near the end of the [puzzle page](https://www.cairo-lang.org/the-cairo-games/puzzles2/)).
It essentially computes

$$
h(x,y) = S + [x \bmod 2^{248}] P_0 + [x >> 248] P_1 + [y \bmod 2^{248}] P_3 + [y >> 248] P_4,
$$

where $S, P_0, P_1, P_2, P_3$ are elliptic curve points.

Since everything is slower in Cairo-land, we can use `compute_hash_chain(data)` function from [starkware/cairo/common/hash_chain.py](https://github.com/starkware-libs/cairo-lang/blob/master/src/starkware/cairo/common/hash_chain.py) to compute hashes in Python directly.

The two functions are not entirely the same, since the Cairo version assumes that the first element of the array is also its length, whereas the Python one simply computes

`h(data[0], h(data[1], h(..., h(data[n-2], data[n-1]))))`.

Recalling that `initial_value[0] == 48`, we can probably assume that `len(initial_value) == 49`.

Moreover, the first hash is the result of computing:

`FULL_HASH = h(48, h(initial_value[1], h(initial_value[2], h(..., h(initial_value[48], initial_value[49]))...)))`

We might also guess that the values in `initial_value` are all less than 48, otherwise the subsequent calls to `hash_chain()` would overflow.

For now, that means there are $48^{48} \approx 2^{268}$ which is definitely intractable.

### 1.6 `verify()`

```js
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

The `verify()` function is another recursive function which iterates on both `squashed_dict` and `initial_value`,
for $n = 0, 1, \ldots, 5$.

As we might have guessed by now, the `initial_value` array provided by `get_initial_value()` is actually an array representing the starting `state`.

Therefore, we can confirm that the size of `initial_value` must be 48 as well.

Looking at the following Python code should give a better understanding of what this function is actually verifying.

```python
FINAL_STATE_VALUES = [ i // 8 for i in range(48)]

def verify(squashed_dict: list[(int, int, int)], initial_value: list[int]):
    assert len(squashed_dict) == len(initial_value) == 48, "squashed_dict and initial_value must contain 48 elements"
    for idx in range(48):
        n = idx//8
        key, prev_value, new_value = squashed_dict[idx]
        assert n == FINAL_STATE_VALUES[key], "FINAL_STATE_VALUES is not correct"
        assert key == idx, f"squashed_dict is not sorted at key {i}"
        assert prev_value == initial_value[key], f"prev_value for key {key} does not correspond with initial_value"
        assert new_value == FINAL_STATE_VALUES[key], f"prev_value for key {key} does not correspond with initial_value"



# verify that our verify function is working correctly
# we create squashed_dict which is already the final state
fake_squashed_dict = [(i, v, v) for i, v in enumerate(FINAL_STATE_VALUES)]
verify(fake_squashed_dict, FINAL_STATE_VALUES)

# print the 48 values that make up the final state
print(FINAL_STATE_VALUES)
```

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]

The above snippet contained the last clue we are going to extract from the source code.

Up until now, we knew the keys in our `state` were the integers in $\{ 0, 1, \ldots, 47\}$, but we had little information about what values they held.

By looking at the final expected state, we realize that the values are simply integers between 0 and 5 (which all appear exactly 8 times).
Therefore, the same must be true for the elements of `initial_value` since they correspond to a permutation of the final state!

This really restricts the possible outputs of `get_initial_value()`, which means we might actually be able to bruteforce something!

## 2. Solving

### 2.1 Breaking `get_initial_value()`

```python
import itertools

from starkware.cairo.common.hash_chain import compute_hash_chain
from starkware.crypto.signature.fast_pedersen_hash import pedersen_hash
```

We can now start seriously attacking this problem head-on!
Our goal in this section is to intelligently bruteforce the array returned by `get_initial_value()`.
Here's what we have figured out:

1. At the start of the function `initial_value` is an array of length 49, and `initial_value[0] == 48`.
2. The array returned is `initial_value[1:]`, and it contains only numbers in $\{ 0, 1, \ldots, 5 \}$.
3. More specifically, each number in $\{ 0, 1, \ldots, 5 \}$ appears exactly 8 times in `initial_value[1:]`

Let's start by formatting the problem's data into Python.

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

The `data` array must contain only integers from $\{ 0, 1, \ldots, 5 \}$ and `data[0]` defines how many elements to hash.
In other words, `data` is an element from the following set:

$$
\{ ( 0 ), ( 1, i_{1,1} ), ( 2, i_{2,1}, i_{2,2} ), ( 3, i_{3,1}, i_{3,2}, i_{3,3} ), ( 4, i_{4,1}, i_{4,2}, i_{4,3}, i_{4,4} ) ,( 5, i_{5,1}, i_{5,2}, i_{5,3}, i_{5,4}, i_{5,5} ) \, |\,  i_{j,k}  \in \{ 0, 1, \ldots, 5 \} \}
$$

A simple calculation reveals that there are exactly 9331 possiblities, a very reasonable amount of iteration we are willing to sacrifice our CPU cycles for.

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
    h = compute_hash_chain(preimage, hash_func=pedersen_hash)
    if h in hashes:
        print(f"found preimage \t{h}:\t{preimage}")
        hash_preimages[h] = preimage
        if len(hash_preimages) == len(hashes):
            print("\nFound all preimages for the hashes in GIVEN_HASHES!\n")
            break

# update working_values with the preimages we found
for idx, h in GIVEN_HASHES.items():
    preimage = hash_preimages[h]
    for i, val in enumerate(preimage):
        working_values[idx+i] = val

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
print(f"We are missing {len(missing_values)} values: {missing_values} at indices {missing_idx}\n")
```

    found preimage 	1501038259288321437961590173137394957125779122158200548115283728521438213428:	(2, 2, 5)
    found preimage 	1039623306816876893268944011668782810398555904667703809415056949499773381189:	(2, 3, 4)
    found preimage 	1035226353110224801512289478587695122129015832153304072590365512606504328818:	(3, 2, 2, 1)
    found preimage 	196997208112053944281778155212956924860955084720008751336605214240056455402:	(4, 0, 1, 5, 5)
    found preimage 	2245701625176425331085101334837624242646502129018701371434984384296915870715:	(4, 0, 4, 1, 0)
    found preimage 	3560520899812162122215526869789497390123010766571927682749531967294685134040:	(4, 0, 5, 4, 2)
    found preimage 	3537881782324467737440957567711773328493014027685577879465936840743865613662:	(4, 1, 1, 3, 2)
    found preimage 	2508528289207660435870821551803296739495662639464901004905339054353214007301:	(4, 3, 0, 5, 5)
    found preimage 	1508108551069464286813785297355641266663485599320848393798932455588476865295:	(5, 0, 0, 3, 3, 1)

    Found all preimages for the hashes in GIVEN_HASHES!

    We are missing 7 values: [0, 1, 1, 2, 3, 3, 5] at indices [17, 23, 28, 29, 33, 47, 48]

    CPU times: user 8.67 s, sys: 36.9 ms, total: 8.7 s
    Wall time: 8.81 s

We were able to find pre-images for all the hashes, but unfortunately we are still missing 7 values in the `initial_value` array.
Since we know how many times each number must appear in the array, we deduce that the missing values are $(0, 1, 1, 2, 3, 3, 5)$.

The last step will consist in trying all $7!$ (5040) permutation of these values at all the missing indices of `initial_value`.
While this is half the number of iterations required for breaking all hashes in `GIVEN_HASHES`,
each call to `compute_hash_chain(initial_value)` now performs 48 different hashes.

With our fingers crossed, we run the final bruteforce.

```python
%%time

for possibility in itertools.permutations(missing_values):
    for i, idx in enumerate(missing_idx):
        working_values[idx] = possibility[i]
    h = compute_hash_chain(working_values, hash_func=pedersen_hash)
    if h == FULL_HASH:
        break

print("FINISHED\ninitial_value = ", working_values[1:],"\n")
```

    FINISHED
    initial_value =  [5, 0, 0, 3, 3, 1, 4, 0, 4, 1, 0, 4, 0, 5, 4, 2, 3, 4, 0, 1, 5, 5, 1, 3, 2, 2, 1, 0, 1, 2, 2, 5, 2, 4, 1, 1, 3, 2, 2, 3, 4, 4, 3, 0, 5, 5, 5, 3]

    CPU times: user 3min 50s, sys: 495 ms, total: 3min 51s
    Wall time: 3min 51s

Success!

We are able to find the whole `initial_value` array in less than 5 minutes (along with a full day to come up with the solution).

Let's verify that everything is correct:

```python
INITIAL_VALUES = [5, 0, 0, 3, 3, 1, 4, 0, 4, 1, 0, 4, 0, 5, 4, 2, 3, 4, 0, 1, 5, 5, 1, 3, 2, 2, 1, 0, 1, 2, 2, 5, 2, 4, 1, 1, 3, 2, 2, 3, 4, 4, 3, 0, 5, 5, 5, 3]

FULL_INITIAL_VALUES = [48] + INITIAL_VALUES
assert compute_hash_chain(FULL_INITIAL_VALUES) == FULL_HASH, "first hash is incorrect"
for idx, h in GIVEN_HASHES.items():
    num_vals = FULL_INITIAL_VALUES[idx]
    hash_block = FULL_INITIAL_VALUES[idx:idx+1+num_vals]
    assert compute_hash_chain(hash_block) == h, f"hash at index {idx} is incorrect"

print("all good!")
```

    all good!

### 2.2 Analyzing `run()`

Now that we have the initial values of the `state` disctionaly, we need to find the actual solution array which solves the problem.

Again, we'll reiterate some of the facts we learned up to now:

1. The solution constists of a sequence of numbers in $\{ 0, 1, \ldots, 5\}$
2. Each of these elements references one of the 6 permutations that can be applied to the current state.
3. Each permuations has order 4, and affect exactly 20 of the 48 elements.

This step will involve analyzing data, so we'll import the usual Python data science stuff.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None) # make sure we can see all 48 element in a row
```

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

We start by looking at what elements are actually affected by each permutation.

In the following matrix, each row defines a permutation of $\{0, 1, \ldots, 47\}$,
and we color the elements in red if the element is moved, and green if it is left intact.

```python
def color_identity(row):
    return [f"background-color: {'red' if row[i] != i else 'green'}" for i in range(len(row))]

display(PERMUTATIONS_DF.style\
    .set_caption('Fixed points of the permutations. 🟩 = fixed, 🟥 = moved')\
    .apply(color_identity,axis=1))
```

<style  type="text/css" >
#T_69b58_row0_col0,#T_69b58_row0_col1,#T_69b58_row0_col2,#T_69b58_row0_col3,#T_69b58_row0_col4,#T_69b58_row0_col5,#T_69b58_row0_col6,#T_69b58_row0_col7,#T_69b58_row0_col8,#T_69b58_row0_col9,#T_69b58_row0_col10,#T_69b58_row0_col16,#T_69b58_row0_col17,#T_69b58_row0_col18,#T_69b58_row0_col24,#T_69b58_row0_col25,#T_69b58_row0_col26,#T_69b58_row0_col32,#T_69b58_row0_col33,#T_69b58_row0_col34,#T_69b58_row1_col2,#T_69b58_row1_col3,#T_69b58_row1_col4,#T_69b58_row1_col18,#T_69b58_row1_col19,#T_69b58_row1_col20,#T_69b58_row1_col24,#T_69b58_row1_col25,#T_69b58_row1_col26,#T_69b58_row1_col27,#T_69b58_row1_col28,#T_69b58_row1_col29,#T_69b58_row1_col30,#T_69b58_row1_col31,#T_69b58_row1_col32,#T_69b58_row1_col38,#T_69b58_row1_col39,#T_69b58_row1_col42,#T_69b58_row1_col43,#T_69b58_row1_col44,#T_69b58_row2_col4,#T_69b58_row2_col5,#T_69b58_row2_col6,#T_69b58_row2_col10,#T_69b58_row2_col11,#T_69b58_row2_col12,#T_69b58_row2_col16,#T_69b58_row2_col17,#T_69b58_row2_col18,#T_69b58_row2_col19,#T_69b58_row2_col20,#T_69b58_row2_col21,#T_69b58_row2_col22,#T_69b58_row2_col23,#T_69b58_row2_col24,#T_69b58_row2_col30,#T_69b58_row2_col31,#T_69b58_row2_col40,#T_69b58_row2_col41,#T_69b58_row2_col42,#T_69b58_row3_col0,#T_69b58_row3_col6,#T_69b58_row3_col7,#T_69b58_row3_col8,#T_69b58_row3_col9,#T_69b58_row3_col10,#T_69b58_row3_col11,#T_69b58_row3_col12,#T_69b58_row3_col13,#T_69b58_row3_col14,#T_69b58_row3_col15,#T_69b58_row3_col16,#T_69b58_row3_col22,#T_69b58_row3_col23,#T_69b58_row3_col34,#T_69b58_row3_col35,#T_69b58_row3_col36,#T_69b58_row3_col40,#T_69b58_row3_col46,#T_69b58_row3_col47,#T_69b58_row4_col0,#T_69b58_row4_col1,#T_69b58_row4_col2,#T_69b58_row4_col8,#T_69b58_row4_col14,#T_69b58_row4_col15,#T_69b58_row4_col26,#T_69b58_row4_col27,#T_69b58_row4_col28,#T_69b58_row4_col32,#T_69b58_row4_col33,#T_69b58_row4_col34,#T_69b58_row4_col35,#T_69b58_row4_col36,#T_69b58_row4_col37,#T_69b58_row4_col38,#T_69b58_row4_col39,#T_69b58_row4_col44,#T_69b58_row4_col45,#T_69b58_row4_col46,#T_69b58_row5_col12,#T_69b58_row5_col13,#T_69b58_row5_col14,#T_69b58_row5_col20,#T_69b58_row5_col21,#T_69b58_row5_col22,#T_69b58_row5_col28,#T_69b58_row5_col29,#T_69b58_row5_col30,#T_69b58_row5_col36,#T_69b58_row5_col37,#T_69b58_row5_col38,#T_69b58_row5_col40,#T_69b58_row5_col41,#T_69b58_row5_col42,#T_69b58_row5_col43,#T_69b58_row5_col44,#T_69b58_row5_col45,#T_69b58_row5_col46,#T_69b58_row5_col47{
            background-color:  red;
        }#T_69b58_row0_col11,#T_69b58_row0_col12,#T_69b58_row0_col13,#T_69b58_row0_col14,#T_69b58_row0_col15,#T_69b58_row0_col19,#T_69b58_row0_col20,#T_69b58_row0_col21,#T_69b58_row0_col22,#T_69b58_row0_col23,#T_69b58_row0_col27,#T_69b58_row0_col28,#T_69b58_row0_col29,#T_69b58_row0_col30,#T_69b58_row0_col31,#T_69b58_row0_col35,#T_69b58_row0_col36,#T_69b58_row0_col37,#T_69b58_row0_col38,#T_69b58_row0_col39,#T_69b58_row0_col40,#T_69b58_row0_col41,#T_69b58_row0_col42,#T_69b58_row0_col43,#T_69b58_row0_col44,#T_69b58_row0_col45,#T_69b58_row0_col46,#T_69b58_row0_col47,#T_69b58_row1_col0,#T_69b58_row1_col1,#T_69b58_row1_col5,#T_69b58_row1_col6,#T_69b58_row1_col7,#T_69b58_row1_col8,#T_69b58_row1_col9,#T_69b58_row1_col10,#T_69b58_row1_col11,#T_69b58_row1_col12,#T_69b58_row1_col13,#T_69b58_row1_col14,#T_69b58_row1_col15,#T_69b58_row1_col16,#T_69b58_row1_col17,#T_69b58_row1_col21,#T_69b58_row1_col22,#T_69b58_row1_col23,#T_69b58_row1_col33,#T_69b58_row1_col34,#T_69b58_row1_col35,#T_69b58_row1_col36,#T_69b58_row1_col37,#T_69b58_row1_col40,#T_69b58_row1_col41,#T_69b58_row1_col45,#T_69b58_row1_col46,#T_69b58_row1_col47,#T_69b58_row2_col0,#T_69b58_row2_col1,#T_69b58_row2_col2,#T_69b58_row2_col3,#T_69b58_row2_col7,#T_69b58_row2_col8,#T_69b58_row2_col9,#T_69b58_row2_col13,#T_69b58_row2_col14,#T_69b58_row2_col15,#T_69b58_row2_col25,#T_69b58_row2_col26,#T_69b58_row2_col27,#T_69b58_row2_col28,#T_69b58_row2_col29,#T_69b58_row2_col32,#T_69b58_row2_col33,#T_69b58_row2_col34,#T_69b58_row2_col35,#T_69b58_row2_col36,#T_69b58_row2_col37,#T_69b58_row2_col38,#T_69b58_row2_col39,#T_69b58_row2_col43,#T_69b58_row2_col44,#T_69b58_row2_col45,#T_69b58_row2_col46,#T_69b58_row2_col47,#T_69b58_row3_col1,#T_69b58_row3_col2,#T_69b58_row3_col3,#T_69b58_row3_col4,#T_69b58_row3_col5,#T_69b58_row3_col17,#T_69b58_row3_col18,#T_69b58_row3_col19,#T_69b58_row3_col20,#T_69b58_row3_col21,#T_69b58_row3_col24,#T_69b58_row3_col25,#T_69b58_row3_col26,#T_69b58_row3_col27,#T_69b58_row3_col28,#T_69b58_row3_col29,#T_69b58_row3_col30,#T_69b58_row3_col31,#T_69b58_row3_col32,#T_69b58_row3_col33,#T_69b58_row3_col37,#T_69b58_row3_col38,#T_69b58_row3_col39,#T_69b58_row3_col41,#T_69b58_row3_col42,#T_69b58_row3_col43,#T_69b58_row3_col44,#T_69b58_row3_col45,#T_69b58_row4_col3,#T_69b58_row4_col4,#T_69b58_row4_col5,#T_69b58_row4_col6,#T_69b58_row4_col7,#T_69b58_row4_col9,#T_69b58_row4_col10,#T_69b58_row4_col11,#T_69b58_row4_col12,#T_69b58_row4_col13,#T_69b58_row4_col16,#T_69b58_row4_col17,#T_69b58_row4_col18,#T_69b58_row4_col19,#T_69b58_row4_col20,#T_69b58_row4_col21,#T_69b58_row4_col22,#T_69b58_row4_col23,#T_69b58_row4_col24,#T_69b58_row4_col25,#T_69b58_row4_col29,#T_69b58_row4_col30,#T_69b58_row4_col31,#T_69b58_row4_col40,#T_69b58_row4_col41,#T_69b58_row4_col42,#T_69b58_row4_col43,#T_69b58_row4_col47,#T_69b58_row5_col0,#T_69b58_row5_col1,#T_69b58_row5_col2,#T_69b58_row5_col3,#T_69b58_row5_col4,#T_69b58_row5_col5,#T_69b58_row5_col6,#T_69b58_row5_col7,#T_69b58_row5_col8,#T_69b58_row5_col9,#T_69b58_row5_col10,#T_69b58_row5_col11,#T_69b58_row5_col15,#T_69b58_row5_col16,#T_69b58_row5_col17,#T_69b58_row5_col18,#T_69b58_row5_col19,#T_69b58_row5_col23,#T_69b58_row5_col24,#T_69b58_row5_col25,#T_69b58_row5_col26,#T_69b58_row5_col27,#T_69b58_row5_col31,#T_69b58_row5_col32,#T_69b58_row5_col33,#T_69b58_row5_col34,#T_69b58_row5_col35,#T_69b58_row5_col39{
            background-color:  green;
        }</style>
<table id="T_69b58_" ><caption>Fixed points of the permutations. 🟩 = fixed, 🟥 = moved</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>        <th class="col_heading level0 col18" >18</th>        <th class="col_heading level0 col19" >19</th>        <th class="col_heading level0 col20" >20</th>        <th class="col_heading level0 col21" >21</th>        <th class="col_heading level0 col22" >22</th>        <th class="col_heading level0 col23" >23</th>        <th class="col_heading level0 col24" >24</th>        <th class="col_heading level0 col25" >25</th>        <th class="col_heading level0 col26" >26</th>        <th class="col_heading level0 col27" >27</th>        <th class="col_heading level0 col28" >28</th>        <th class="col_heading level0 col29" >29</th>        <th class="col_heading level0 col30" >30</th>        <th class="col_heading level0 col31" >31</th>        <th class="col_heading level0 col32" >32</th>        <th class="col_heading level0 col33" >33</th>        <th class="col_heading level0 col34" >34</th>        <th class="col_heading level0 col35" >35</th>        <th class="col_heading level0 col36" >36</th>        <th class="col_heading level0 col37" >37</th>        <th class="col_heading level0 col38" >38</th>        <th class="col_heading level0 col39" >39</th>        <th class="col_heading level0 col40" >40</th>        <th class="col_heading level0 col41" >41</th>        <th class="col_heading level0 col42" >42</th>        <th class="col_heading level0 col43" >43</th>        <th class="col_heading level0 col44" >44</th>        <th class="col_heading level0 col45" >45</th>        <th class="col_heading level0 col46" >46</th>        <th class="col_heading level0 col47" >47</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_69b58_level0_row0" class="row_heading level0 row0" >U</th>
                        <td id="T_69b58_row0_col0" class="data row0 col0" >6</td>
                        <td id="T_69b58_row0_col1" class="data row0 col1" >7</td>
                        <td id="T_69b58_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_69b58_row0_col3" class="data row0 col3" >1</td>
                        <td id="T_69b58_row0_col4" class="data row0 col4" >2</td>
                        <td id="T_69b58_row0_col5" class="data row0 col5" >3</td>
                        <td id="T_69b58_row0_col6" class="data row0 col6" >4</td>
                        <td id="T_69b58_row0_col7" class="data row0 col7" >5</td>
                        <td id="T_69b58_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_69b58_row0_col9" class="data row0 col9" >17</td>
                        <td id="T_69b58_row0_col10" class="data row0 col10" >18</td>
                        <td id="T_69b58_row0_col11" class="data row0 col11" >11</td>
                        <td id="T_69b58_row0_col12" class="data row0 col12" >12</td>
                        <td id="T_69b58_row0_col13" class="data row0 col13" >13</td>
                        <td id="T_69b58_row0_col14" class="data row0 col14" >14</td>
                        <td id="T_69b58_row0_col15" class="data row0 col15" >15</td>
                        <td id="T_69b58_row0_col16" class="data row0 col16" >24</td>
                        <td id="T_69b58_row0_col17" class="data row0 col17" >25</td>
                        <td id="T_69b58_row0_col18" class="data row0 col18" >26</td>
                        <td id="T_69b58_row0_col19" class="data row0 col19" >19</td>
                        <td id="T_69b58_row0_col20" class="data row0 col20" >20</td>
                        <td id="T_69b58_row0_col21" class="data row0 col21" >21</td>
                        <td id="T_69b58_row0_col22" class="data row0 col22" >22</td>
                        <td id="T_69b58_row0_col23" class="data row0 col23" >23</td>
                        <td id="T_69b58_row0_col24" class="data row0 col24" >32</td>
                        <td id="T_69b58_row0_col25" class="data row0 col25" >33</td>
                        <td id="T_69b58_row0_col26" class="data row0 col26" >34</td>
                        <td id="T_69b58_row0_col27" class="data row0 col27" >27</td>
                        <td id="T_69b58_row0_col28" class="data row0 col28" >28</td>
                        <td id="T_69b58_row0_col29" class="data row0 col29" >29</td>
                        <td id="T_69b58_row0_col30" class="data row0 col30" >30</td>
                        <td id="T_69b58_row0_col31" class="data row0 col31" >31</td>
                        <td id="T_69b58_row0_col32" class="data row0 col32" >8</td>
                        <td id="T_69b58_row0_col33" class="data row0 col33" >9</td>
                        <td id="T_69b58_row0_col34" class="data row0 col34" >10</td>
                        <td id="T_69b58_row0_col35" class="data row0 col35" >35</td>
                        <td id="T_69b58_row0_col36" class="data row0 col36" >36</td>
                        <td id="T_69b58_row0_col37" class="data row0 col37" >37</td>
                        <td id="T_69b58_row0_col38" class="data row0 col38" >38</td>
                        <td id="T_69b58_row0_col39" class="data row0 col39" >39</td>
                        <td id="T_69b58_row0_col40" class="data row0 col40" >40</td>
                        <td id="T_69b58_row0_col41" class="data row0 col41" >41</td>
                        <td id="T_69b58_row0_col42" class="data row0 col42" >42</td>
                        <td id="T_69b58_row0_col43" class="data row0 col43" >43</td>
                        <td id="T_69b58_row0_col44" class="data row0 col44" >44</td>
                        <td id="T_69b58_row0_col45" class="data row0 col45" >45</td>
                        <td id="T_69b58_row0_col46" class="data row0 col46" >46</td>
                        <td id="T_69b58_row0_col47" class="data row0 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_69b58_level0_row1" class="row_heading level0 row1" >R</th>
                        <td id="T_69b58_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_69b58_row1_col1" class="data row1 col1" >1</td>
                        <td id="T_69b58_row1_col2" class="data row1 col2" >18</td>
                        <td id="T_69b58_row1_col3" class="data row1 col3" >19</td>
                        <td id="T_69b58_row1_col4" class="data row1 col4" >20</td>
                        <td id="T_69b58_row1_col5" class="data row1 col5" >5</td>
                        <td id="T_69b58_row1_col6" class="data row1 col6" >6</td>
                        <td id="T_69b58_row1_col7" class="data row1 col7" >7</td>
                        <td id="T_69b58_row1_col8" class="data row1 col8" >8</td>
                        <td id="T_69b58_row1_col9" class="data row1 col9" >9</td>
                        <td id="T_69b58_row1_col10" class="data row1 col10" >10</td>
                        <td id="T_69b58_row1_col11" class="data row1 col11" >11</td>
                        <td id="T_69b58_row1_col12" class="data row1 col12" >12</td>
                        <td id="T_69b58_row1_col13" class="data row1 col13" >13</td>
                        <td id="T_69b58_row1_col14" class="data row1 col14" >14</td>
                        <td id="T_69b58_row1_col15" class="data row1 col15" >15</td>
                        <td id="T_69b58_row1_col16" class="data row1 col16" >16</td>
                        <td id="T_69b58_row1_col17" class="data row1 col17" >17</td>
                        <td id="T_69b58_row1_col18" class="data row1 col18" >42</td>
                        <td id="T_69b58_row1_col19" class="data row1 col19" >43</td>
                        <td id="T_69b58_row1_col20" class="data row1 col20" >44</td>
                        <td id="T_69b58_row1_col21" class="data row1 col21" >21</td>
                        <td id="T_69b58_row1_col22" class="data row1 col22" >22</td>
                        <td id="T_69b58_row1_col23" class="data row1 col23" >23</td>
                        <td id="T_69b58_row1_col24" class="data row1 col24" >30</td>
                        <td id="T_69b58_row1_col25" class="data row1 col25" >31</td>
                        <td id="T_69b58_row1_col26" class="data row1 col26" >24</td>
                        <td id="T_69b58_row1_col27" class="data row1 col27" >25</td>
                        <td id="T_69b58_row1_col28" class="data row1 col28" >26</td>
                        <td id="T_69b58_row1_col29" class="data row1 col29" >27</td>
                        <td id="T_69b58_row1_col30" class="data row1 col30" >28</td>
                        <td id="T_69b58_row1_col31" class="data row1 col31" >29</td>
                        <td id="T_69b58_row1_col32" class="data row1 col32" >4</td>
                        <td id="T_69b58_row1_col33" class="data row1 col33" >33</td>
                        <td id="T_69b58_row1_col34" class="data row1 col34" >34</td>
                        <td id="T_69b58_row1_col35" class="data row1 col35" >35</td>
                        <td id="T_69b58_row1_col36" class="data row1 col36" >36</td>
                        <td id="T_69b58_row1_col37" class="data row1 col37" >37</td>
                        <td id="T_69b58_row1_col38" class="data row1 col38" >2</td>
                        <td id="T_69b58_row1_col39" class="data row1 col39" >3</td>
                        <td id="T_69b58_row1_col40" class="data row1 col40" >40</td>
                        <td id="T_69b58_row1_col41" class="data row1 col41" >41</td>
                        <td id="T_69b58_row1_col42" class="data row1 col42" >38</td>
                        <td id="T_69b58_row1_col43" class="data row1 col43" >39</td>
                        <td id="T_69b58_row1_col44" class="data row1 col44" >32</td>
                        <td id="T_69b58_row1_col45" class="data row1 col45" >45</td>
                        <td id="T_69b58_row1_col46" class="data row1 col46" >46</td>
                        <td id="T_69b58_row1_col47" class="data row1 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_69b58_level0_row2" class="row_heading level0 row2" >F</th>
                        <td id="T_69b58_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_69b58_row2_col1" class="data row2 col1" >1</td>
                        <td id="T_69b58_row2_col2" class="data row2 col2" >2</td>
                        <td id="T_69b58_row2_col3" class="data row2 col3" >3</td>
                        <td id="T_69b58_row2_col4" class="data row2 col4" >10</td>
                        <td id="T_69b58_row2_col5" class="data row2 col5" >11</td>
                        <td id="T_69b58_row2_col6" class="data row2 col6" >12</td>
                        <td id="T_69b58_row2_col7" class="data row2 col7" >7</td>
                        <td id="T_69b58_row2_col8" class="data row2 col8" >8</td>
                        <td id="T_69b58_row2_col9" class="data row2 col9" >9</td>
                        <td id="T_69b58_row2_col10" class="data row2 col10" >40</td>
                        <td id="T_69b58_row2_col11" class="data row2 col11" >41</td>
                        <td id="T_69b58_row2_col12" class="data row2 col12" >42</td>
                        <td id="T_69b58_row2_col13" class="data row2 col13" >13</td>
                        <td id="T_69b58_row2_col14" class="data row2 col14" >14</td>
                        <td id="T_69b58_row2_col15" class="data row2 col15" >15</td>
                        <td id="T_69b58_row2_col16" class="data row2 col16" >22</td>
                        <td id="T_69b58_row2_col17" class="data row2 col17" >23</td>
                        <td id="T_69b58_row2_col18" class="data row2 col18" >16</td>
                        <td id="T_69b58_row2_col19" class="data row2 col19" >17</td>
                        <td id="T_69b58_row2_col20" class="data row2 col20" >18</td>
                        <td id="T_69b58_row2_col21" class="data row2 col21" >19</td>
                        <td id="T_69b58_row2_col22" class="data row2 col22" >20</td>
                        <td id="T_69b58_row2_col23" class="data row2 col23" >21</td>
                        <td id="T_69b58_row2_col24" class="data row2 col24" >6</td>
                        <td id="T_69b58_row2_col25" class="data row2 col25" >25</td>
                        <td id="T_69b58_row2_col26" class="data row2 col26" >26</td>
                        <td id="T_69b58_row2_col27" class="data row2 col27" >27</td>
                        <td id="T_69b58_row2_col28" class="data row2 col28" >28</td>
                        <td id="T_69b58_row2_col29" class="data row2 col29" >29</td>
                        <td id="T_69b58_row2_col30" class="data row2 col30" >4</td>
                        <td id="T_69b58_row2_col31" class="data row2 col31" >5</td>
                        <td id="T_69b58_row2_col32" class="data row2 col32" >32</td>
                        <td id="T_69b58_row2_col33" class="data row2 col33" >33</td>
                        <td id="T_69b58_row2_col34" class="data row2 col34" >34</td>
                        <td id="T_69b58_row2_col35" class="data row2 col35" >35</td>
                        <td id="T_69b58_row2_col36" class="data row2 col36" >36</td>
                        <td id="T_69b58_row2_col37" class="data row2 col37" >37</td>
                        <td id="T_69b58_row2_col38" class="data row2 col38" >38</td>
                        <td id="T_69b58_row2_col39" class="data row2 col39" >39</td>
                        <td id="T_69b58_row2_col40" class="data row2 col40" >30</td>
                        <td id="T_69b58_row2_col41" class="data row2 col41" >31</td>
                        <td id="T_69b58_row2_col42" class="data row2 col42" >24</td>
                        <td id="T_69b58_row2_col43" class="data row2 col43" >43</td>
                        <td id="T_69b58_row2_col44" class="data row2 col44" >44</td>
                        <td id="T_69b58_row2_col45" class="data row2 col45" >45</td>
                        <td id="T_69b58_row2_col46" class="data row2 col46" >46</td>
                        <td id="T_69b58_row2_col47" class="data row2 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_69b58_level0_row3" class="row_heading level0 row3" >L</th>
                        <td id="T_69b58_row3_col0" class="data row3 col0" >36</td>
                        <td id="T_69b58_row3_col1" class="data row3 col1" >1</td>
                        <td id="T_69b58_row3_col2" class="data row3 col2" >2</td>
                        <td id="T_69b58_row3_col3" class="data row3 col3" >3</td>
                        <td id="T_69b58_row3_col4" class="data row3 col4" >4</td>
                        <td id="T_69b58_row3_col5" class="data row3 col5" >5</td>
                        <td id="T_69b58_row3_col6" class="data row3 col6" >34</td>
                        <td id="T_69b58_row3_col7" class="data row3 col7" >35</td>
                        <td id="T_69b58_row3_col8" class="data row3 col8" >14</td>
                        <td id="T_69b58_row3_col9" class="data row3 col9" >15</td>
                        <td id="T_69b58_row3_col10" class="data row3 col10" >8</td>
                        <td id="T_69b58_row3_col11" class="data row3 col11" >9</td>
                        <td id="T_69b58_row3_col12" class="data row3 col12" >10</td>
                        <td id="T_69b58_row3_col13" class="data row3 col13" >11</td>
                        <td id="T_69b58_row3_col14" class="data row3 col14" >12</td>
                        <td id="T_69b58_row3_col15" class="data row3 col15" >13</td>
                        <td id="T_69b58_row3_col16" class="data row3 col16" >0</td>
                        <td id="T_69b58_row3_col17" class="data row3 col17" >17</td>
                        <td id="T_69b58_row3_col18" class="data row3 col18" >18</td>
                        <td id="T_69b58_row3_col19" class="data row3 col19" >19</td>
                        <td id="T_69b58_row3_col20" class="data row3 col20" >20</td>
                        <td id="T_69b58_row3_col21" class="data row3 col21" >21</td>
                        <td id="T_69b58_row3_col22" class="data row3 col22" >6</td>
                        <td id="T_69b58_row3_col23" class="data row3 col23" >7</td>
                        <td id="T_69b58_row3_col24" class="data row3 col24" >24</td>
                        <td id="T_69b58_row3_col25" class="data row3 col25" >25</td>
                        <td id="T_69b58_row3_col26" class="data row3 col26" >26</td>
                        <td id="T_69b58_row3_col27" class="data row3 col27" >27</td>
                        <td id="T_69b58_row3_col28" class="data row3 col28" >28</td>
                        <td id="T_69b58_row3_col29" class="data row3 col29" >29</td>
                        <td id="T_69b58_row3_col30" class="data row3 col30" >30</td>
                        <td id="T_69b58_row3_col31" class="data row3 col31" >31</td>
                        <td id="T_69b58_row3_col32" class="data row3 col32" >32</td>
                        <td id="T_69b58_row3_col33" class="data row3 col33" >33</td>
                        <td id="T_69b58_row3_col34" class="data row3 col34" >46</td>
                        <td id="T_69b58_row3_col35" class="data row3 col35" >47</td>
                        <td id="T_69b58_row3_col36" class="data row3 col36" >40</td>
                        <td id="T_69b58_row3_col37" class="data row3 col37" >37</td>
                        <td id="T_69b58_row3_col38" class="data row3 col38" >38</td>
                        <td id="T_69b58_row3_col39" class="data row3 col39" >39</td>
                        <td id="T_69b58_row3_col40" class="data row3 col40" >16</td>
                        <td id="T_69b58_row3_col41" class="data row3 col41" >41</td>
                        <td id="T_69b58_row3_col42" class="data row3 col42" >42</td>
                        <td id="T_69b58_row3_col43" class="data row3 col43" >43</td>
                        <td id="T_69b58_row3_col44" class="data row3 col44" >44</td>
                        <td id="T_69b58_row3_col45" class="data row3 col45" >45</td>
                        <td id="T_69b58_row3_col46" class="data row3 col46" >22</td>
                        <td id="T_69b58_row3_col47" class="data row3 col47" >23</td>
            </tr>
            <tr>
                        <th id="T_69b58_level0_row4" class="row_heading level0 row4" >B</th>
                        <td id="T_69b58_row4_col0" class="data row4 col0" >26</td>
                        <td id="T_69b58_row4_col1" class="data row4 col1" >27</td>
                        <td id="T_69b58_row4_col2" class="data row4 col2" >28</td>
                        <td id="T_69b58_row4_col3" class="data row4 col3" >3</td>
                        <td id="T_69b58_row4_col4" class="data row4 col4" >4</td>
                        <td id="T_69b58_row4_col5" class="data row4 col5" >5</td>
                        <td id="T_69b58_row4_col6" class="data row4 col6" >6</td>
                        <td id="T_69b58_row4_col7" class="data row4 col7" >7</td>
                        <td id="T_69b58_row4_col8" class="data row4 col8" >2</td>
                        <td id="T_69b58_row4_col9" class="data row4 col9" >9</td>
                        <td id="T_69b58_row4_col10" class="data row4 col10" >10</td>
                        <td id="T_69b58_row4_col11" class="data row4 col11" >11</td>
                        <td id="T_69b58_row4_col12" class="data row4 col12" >12</td>
                        <td id="T_69b58_row4_col13" class="data row4 col13" >13</td>
                        <td id="T_69b58_row4_col14" class="data row4 col14" >0</td>
                        <td id="T_69b58_row4_col15" class="data row4 col15" >1</td>
                        <td id="T_69b58_row4_col16" class="data row4 col16" >16</td>
                        <td id="T_69b58_row4_col17" class="data row4 col17" >17</td>
                        <td id="T_69b58_row4_col18" class="data row4 col18" >18</td>
                        <td id="T_69b58_row4_col19" class="data row4 col19" >19</td>
                        <td id="T_69b58_row4_col20" class="data row4 col20" >20</td>
                        <td id="T_69b58_row4_col21" class="data row4 col21" >21</td>
                        <td id="T_69b58_row4_col22" class="data row4 col22" >22</td>
                        <td id="T_69b58_row4_col23" class="data row4 col23" >23</td>
                        <td id="T_69b58_row4_col24" class="data row4 col24" >24</td>
                        <td id="T_69b58_row4_col25" class="data row4 col25" >25</td>
                        <td id="T_69b58_row4_col26" class="data row4 col26" >44</td>
                        <td id="T_69b58_row4_col27" class="data row4 col27" >45</td>
                        <td id="T_69b58_row4_col28" class="data row4 col28" >46</td>
                        <td id="T_69b58_row4_col29" class="data row4 col29" >29</td>
                        <td id="T_69b58_row4_col30" class="data row4 col30" >30</td>
                        <td id="T_69b58_row4_col31" class="data row4 col31" >31</td>
                        <td id="T_69b58_row4_col32" class="data row4 col32" >38</td>
                        <td id="T_69b58_row4_col33" class="data row4 col33" >39</td>
                        <td id="T_69b58_row4_col34" class="data row4 col34" >32</td>
                        <td id="T_69b58_row4_col35" class="data row4 col35" >33</td>
                        <td id="T_69b58_row4_col36" class="data row4 col36" >34</td>
                        <td id="T_69b58_row4_col37" class="data row4 col37" >35</td>
                        <td id="T_69b58_row4_col38" class="data row4 col38" >36</td>
                        <td id="T_69b58_row4_col39" class="data row4 col39" >37</td>
                        <td id="T_69b58_row4_col40" class="data row4 col40" >40</td>
                        <td id="T_69b58_row4_col41" class="data row4 col41" >41</td>
                        <td id="T_69b58_row4_col42" class="data row4 col42" >42</td>
                        <td id="T_69b58_row4_col43" class="data row4 col43" >43</td>
                        <td id="T_69b58_row4_col44" class="data row4 col44" >14</td>
                        <td id="T_69b58_row4_col45" class="data row4 col45" >15</td>
                        <td id="T_69b58_row4_col46" class="data row4 col46" >8</td>
                        <td id="T_69b58_row4_col47" class="data row4 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_69b58_level0_row5" class="row_heading level0 row5" >D</th>
                        <td id="T_69b58_row5_col0" class="data row5 col0" >0</td>
                        <td id="T_69b58_row5_col1" class="data row5 col1" >1</td>
                        <td id="T_69b58_row5_col2" class="data row5 col2" >2</td>
                        <td id="T_69b58_row5_col3" class="data row5 col3" >3</td>
                        <td id="T_69b58_row5_col4" class="data row5 col4" >4</td>
                        <td id="T_69b58_row5_col5" class="data row5 col5" >5</td>
                        <td id="T_69b58_row5_col6" class="data row5 col6" >6</td>
                        <td id="T_69b58_row5_col7" class="data row5 col7" >7</td>
                        <td id="T_69b58_row5_col8" class="data row5 col8" >8</td>
                        <td id="T_69b58_row5_col9" class="data row5 col9" >9</td>
                        <td id="T_69b58_row5_col10" class="data row5 col10" >10</td>
                        <td id="T_69b58_row5_col11" class="data row5 col11" >11</td>
                        <td id="T_69b58_row5_col12" class="data row5 col12" >36</td>
                        <td id="T_69b58_row5_col13" class="data row5 col13" >37</td>
                        <td id="T_69b58_row5_col14" class="data row5 col14" >38</td>
                        <td id="T_69b58_row5_col15" class="data row5 col15" >15</td>
                        <td id="T_69b58_row5_col16" class="data row5 col16" >16</td>
                        <td id="T_69b58_row5_col17" class="data row5 col17" >17</td>
                        <td id="T_69b58_row5_col18" class="data row5 col18" >18</td>
                        <td id="T_69b58_row5_col19" class="data row5 col19" >19</td>
                        <td id="T_69b58_row5_col20" class="data row5 col20" >12</td>
                        <td id="T_69b58_row5_col21" class="data row5 col21" >13</td>
                        <td id="T_69b58_row5_col22" class="data row5 col22" >14</td>
                        <td id="T_69b58_row5_col23" class="data row5 col23" >23</td>
                        <td id="T_69b58_row5_col24" class="data row5 col24" >24</td>
                        <td id="T_69b58_row5_col25" class="data row5 col25" >25</td>
                        <td id="T_69b58_row5_col26" class="data row5 col26" >26</td>
                        <td id="T_69b58_row5_col27" class="data row5 col27" >27</td>
                        <td id="T_69b58_row5_col28" class="data row5 col28" >20</td>
                        <td id="T_69b58_row5_col29" class="data row5 col29" >21</td>
                        <td id="T_69b58_row5_col30" class="data row5 col30" >22</td>
                        <td id="T_69b58_row5_col31" class="data row5 col31" >31</td>
                        <td id="T_69b58_row5_col32" class="data row5 col32" >32</td>
                        <td id="T_69b58_row5_col33" class="data row5 col33" >33</td>
                        <td id="T_69b58_row5_col34" class="data row5 col34" >34</td>
                        <td id="T_69b58_row5_col35" class="data row5 col35" >35</td>
                        <td id="T_69b58_row5_col36" class="data row5 col36" >28</td>
                        <td id="T_69b58_row5_col37" class="data row5 col37" >29</td>
                        <td id="T_69b58_row5_col38" class="data row5 col38" >30</td>
                        <td id="T_69b58_row5_col39" class="data row5 col39" >39</td>
                        <td id="T_69b58_row5_col40" class="data row5 col40" >46</td>
                        <td id="T_69b58_row5_col41" class="data row5 col41" >47</td>
                        <td id="T_69b58_row5_col42" class="data row5 col42" >40</td>
                        <td id="T_69b58_row5_col43" class="data row5 col43" >41</td>
                        <td id="T_69b58_row5_col44" class="data row5 col44" >42</td>
                        <td id="T_69b58_row5_col45" class="data row5 col45" >43</td>
                        <td id="T_69b58_row5_col46" class="data row5 col46" >44</td>
                        <td id="T_69b58_row5_col47" class="data row5 col47" >45</td>
            </tr>
    </tbody></table>

We already knew that each permutation only affects 20 items, but there does seem to be quite a bit more structure that we need to explore.
Let's note the following:

1. Permutations affect big blocks consecutive elements, and leave other large blocks intact.
2. Each row has one big block which is permuted, and several other tinier blocks of size < 4.
3. Most blocks (tiny and large) seem to preserve some order.
4. Groups of 8 columns seem to behave in a very structured way.

These observations by themselves are not so helpful, but together they make the solution incredibly obvious (with hindsight).
What finally made it click was when we used colors to represent each group of 8 columns.

In the following plot, we assigned to each element a color depending on its value divided by 8.
We also color in pink the labels of the elements that change position but stay in the same group.

```python
COLORS = {
    0: 'lightgrey', # ⬜️
    1: 'orange', # 🟧
    2: 'green', # 🟩
    3: 'red', # 🟥
    4: 'blue', # 🟦
    5: 'yellow', # 🟨
    -1: 'magenta' # 🟪
}

def color_group(val):
    return f'background-color: {COLORS[val//BLOCK_SIZE]}'

def color_not_id_in_group(row):
    return [f'color: {COLORS[-1]}' if row[i] != i and row[i] // BLOCK_SIZE == i // BLOCK_SIZE else '' for i in range(len(row))]

display(PERMUTATIONS_DF.style\
    .set_caption('Permutations visualized with "arbitrary" colors. 🟪 = move to same block')\
    .applymap(color_group)\
    .apply(color_not_id_in_group,axis=1))
```

<style  type="text/css" >
#T_b7c21_row0_col0,#T_b7c21_row0_col1,#T_b7c21_row0_col2,#T_b7c21_row0_col3,#T_b7c21_row0_col4,#T_b7c21_row0_col5,#T_b7c21_row0_col6,#T_b7c21_row0_col7{
            background-color:  lightgrey;
            color:  magenta;
        }#T_b7c21_row0_col8,#T_b7c21_row0_col9,#T_b7c21_row0_col10,#T_b7c21_row0_col19,#T_b7c21_row0_col20,#T_b7c21_row0_col21,#T_b7c21_row0_col22,#T_b7c21_row0_col23,#T_b7c21_row1_col2,#T_b7c21_row1_col3,#T_b7c21_row1_col4,#T_b7c21_row1_col16,#T_b7c21_row1_col17,#T_b7c21_row1_col21,#T_b7c21_row1_col22,#T_b7c21_row1_col23,#T_b7c21_row3_col17,#T_b7c21_row3_col18,#T_b7c21_row3_col19,#T_b7c21_row3_col20,#T_b7c21_row3_col21,#T_b7c21_row3_col40,#T_b7c21_row3_col46,#T_b7c21_row3_col47,#T_b7c21_row4_col16,#T_b7c21_row4_col17,#T_b7c21_row4_col18,#T_b7c21_row4_col19,#T_b7c21_row4_col20,#T_b7c21_row4_col21,#T_b7c21_row4_col22,#T_b7c21_row4_col23,#T_b7c21_row5_col16,#T_b7c21_row5_col17,#T_b7c21_row5_col18,#T_b7c21_row5_col19,#T_b7c21_row5_col23,#T_b7c21_row5_col28,#T_b7c21_row5_col29,#T_b7c21_row5_col30{
            background-color:  green;
        }#T_b7c21_row0_col11,#T_b7c21_row0_col12,#T_b7c21_row0_col13,#T_b7c21_row0_col14,#T_b7c21_row0_col15,#T_b7c21_row0_col32,#T_b7c21_row0_col33,#T_b7c21_row0_col34,#T_b7c21_row1_col8,#T_b7c21_row1_col9,#T_b7c21_row1_col10,#T_b7c21_row1_col11,#T_b7c21_row1_col12,#T_b7c21_row1_col13,#T_b7c21_row1_col14,#T_b7c21_row1_col15,#T_b7c21_row2_col4,#T_b7c21_row2_col5,#T_b7c21_row2_col6,#T_b7c21_row2_col8,#T_b7c21_row2_col9,#T_b7c21_row2_col13,#T_b7c21_row2_col14,#T_b7c21_row2_col15,#T_b7c21_row4_col9,#T_b7c21_row4_col10,#T_b7c21_row4_col11,#T_b7c21_row4_col12,#T_b7c21_row4_col13,#T_b7c21_row4_col44,#T_b7c21_row4_col45,#T_b7c21_row4_col46,#T_b7c21_row5_col8,#T_b7c21_row5_col9,#T_b7c21_row5_col10,#T_b7c21_row5_col11,#T_b7c21_row5_col15,#T_b7c21_row5_col20,#T_b7c21_row5_col21,#T_b7c21_row5_col22{
            background-color:  orange;
        }#T_b7c21_row0_col16,#T_b7c21_row0_col17,#T_b7c21_row0_col18,#T_b7c21_row0_col27,#T_b7c21_row0_col28,#T_b7c21_row0_col29,#T_b7c21_row0_col30,#T_b7c21_row0_col31,#T_b7c21_row2_col25,#T_b7c21_row2_col26,#T_b7c21_row2_col27,#T_b7c21_row2_col28,#T_b7c21_row2_col29,#T_b7c21_row2_col40,#T_b7c21_row2_col41,#T_b7c21_row2_col42,#T_b7c21_row3_col24,#T_b7c21_row3_col25,#T_b7c21_row3_col26,#T_b7c21_row3_col27,#T_b7c21_row3_col28,#T_b7c21_row3_col29,#T_b7c21_row3_col30,#T_b7c21_row3_col31,#T_b7c21_row4_col0,#T_b7c21_row4_col1,#T_b7c21_row4_col2,#T_b7c21_row4_col24,#T_b7c21_row4_col25,#T_b7c21_row4_col29,#T_b7c21_row4_col30,#T_b7c21_row4_col31,#T_b7c21_row5_col24,#T_b7c21_row5_col25,#T_b7c21_row5_col26,#T_b7c21_row5_col27,#T_b7c21_row5_col31,#T_b7c21_row5_col36,#T_b7c21_row5_col37,#T_b7c21_row5_col38{
            background-color:  red;
        }#T_b7c21_row0_col24,#T_b7c21_row0_col25,#T_b7c21_row0_col26,#T_b7c21_row0_col35,#T_b7c21_row0_col36,#T_b7c21_row0_col37,#T_b7c21_row0_col38,#T_b7c21_row0_col39,#T_b7c21_row1_col33,#T_b7c21_row1_col34,#T_b7c21_row1_col35,#T_b7c21_row1_col36,#T_b7c21_row1_col37,#T_b7c21_row1_col42,#T_b7c21_row1_col43,#T_b7c21_row1_col44,#T_b7c21_row2_col32,#T_b7c21_row2_col33,#T_b7c21_row2_col34,#T_b7c21_row2_col35,#T_b7c21_row2_col36,#T_b7c21_row2_col37,#T_b7c21_row2_col38,#T_b7c21_row2_col39,#T_b7c21_row3_col0,#T_b7c21_row3_col6,#T_b7c21_row3_col7,#T_b7c21_row3_col32,#T_b7c21_row3_col33,#T_b7c21_row3_col37,#T_b7c21_row3_col38,#T_b7c21_row3_col39,#T_b7c21_row5_col12,#T_b7c21_row5_col13,#T_b7c21_row5_col14,#T_b7c21_row5_col32,#T_b7c21_row5_col33,#T_b7c21_row5_col34,#T_b7c21_row5_col35,#T_b7c21_row5_col39{
            background-color:  blue;
        }#T_b7c21_row0_col40,#T_b7c21_row0_col41,#T_b7c21_row0_col42,#T_b7c21_row0_col43,#T_b7c21_row0_col44,#T_b7c21_row0_col45,#T_b7c21_row0_col46,#T_b7c21_row0_col47,#T_b7c21_row1_col18,#T_b7c21_row1_col19,#T_b7c21_row1_col20,#T_b7c21_row1_col40,#T_b7c21_row1_col41,#T_b7c21_row1_col45,#T_b7c21_row1_col46,#T_b7c21_row1_col47,#T_b7c21_row2_col10,#T_b7c21_row2_col11,#T_b7c21_row2_col12,#T_b7c21_row2_col43,#T_b7c21_row2_col44,#T_b7c21_row2_col45,#T_b7c21_row2_col46,#T_b7c21_row2_col47,#T_b7c21_row3_col34,#T_b7c21_row3_col35,#T_b7c21_row3_col36,#T_b7c21_row3_col41,#T_b7c21_row3_col42,#T_b7c21_row3_col43,#T_b7c21_row3_col44,#T_b7c21_row3_col45,#T_b7c21_row4_col26,#T_b7c21_row4_col27,#T_b7c21_row4_col28,#T_b7c21_row4_col40,#T_b7c21_row4_col41,#T_b7c21_row4_col42,#T_b7c21_row4_col43,#T_b7c21_row4_col47{
            background-color:  yellow;
        }#T_b7c21_row1_col0,#T_b7c21_row1_col1,#T_b7c21_row1_col5,#T_b7c21_row1_col6,#T_b7c21_row1_col7,#T_b7c21_row1_col32,#T_b7c21_row1_col38,#T_b7c21_row1_col39,#T_b7c21_row2_col0,#T_b7c21_row2_col1,#T_b7c21_row2_col2,#T_b7c21_row2_col3,#T_b7c21_row2_col7,#T_b7c21_row2_col24,#T_b7c21_row2_col30,#T_b7c21_row2_col31,#T_b7c21_row3_col1,#T_b7c21_row3_col2,#T_b7c21_row3_col3,#T_b7c21_row3_col4,#T_b7c21_row3_col5,#T_b7c21_row3_col16,#T_b7c21_row3_col22,#T_b7c21_row3_col23,#T_b7c21_row4_col3,#T_b7c21_row4_col4,#T_b7c21_row4_col5,#T_b7c21_row4_col6,#T_b7c21_row4_col7,#T_b7c21_row4_col8,#T_b7c21_row4_col14,#T_b7c21_row4_col15,#T_b7c21_row5_col0,#T_b7c21_row5_col1,#T_b7c21_row5_col2,#T_b7c21_row5_col3,#T_b7c21_row5_col4,#T_b7c21_row5_col5,#T_b7c21_row5_col6,#T_b7c21_row5_col7{
            background-color:  lightgrey;
        }#T_b7c21_row1_col24,#T_b7c21_row1_col25,#T_b7c21_row1_col26,#T_b7c21_row1_col27,#T_b7c21_row1_col28,#T_b7c21_row1_col29,#T_b7c21_row1_col30,#T_b7c21_row1_col31{
            background-color:  red;
            color:  magenta;
        }#T_b7c21_row2_col16,#T_b7c21_row2_col17,#T_b7c21_row2_col18,#T_b7c21_row2_col19,#T_b7c21_row2_col20,#T_b7c21_row2_col21,#T_b7c21_row2_col22,#T_b7c21_row2_col23{
            background-color:  green;
            color:  magenta;
        }#T_b7c21_row3_col8,#T_b7c21_row3_col9,#T_b7c21_row3_col10,#T_b7c21_row3_col11,#T_b7c21_row3_col12,#T_b7c21_row3_col13,#T_b7c21_row3_col14,#T_b7c21_row3_col15{
            background-color:  orange;
            color:  magenta;
        }#T_b7c21_row4_col32,#T_b7c21_row4_col33,#T_b7c21_row4_col34,#T_b7c21_row4_col35,#T_b7c21_row4_col36,#T_b7c21_row4_col37,#T_b7c21_row4_col38,#T_b7c21_row4_col39{
            background-color:  blue;
            color:  magenta;
        }#T_b7c21_row5_col40,#T_b7c21_row5_col41,#T_b7c21_row5_col42,#T_b7c21_row5_col43,#T_b7c21_row5_col44,#T_b7c21_row5_col45,#T_b7c21_row5_col46,#T_b7c21_row5_col47{
            background-color:  yellow;
            color:  magenta;
        }</style>
<table id="T_b7c21_" ><caption>Permutations visualized with "arbitrary" colors. 🟪 = move to same block</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>        <th class="col_heading level0 col18" >18</th>        <th class="col_heading level0 col19" >19</th>        <th class="col_heading level0 col20" >20</th>        <th class="col_heading level0 col21" >21</th>        <th class="col_heading level0 col22" >22</th>        <th class="col_heading level0 col23" >23</th>        <th class="col_heading level0 col24" >24</th>        <th class="col_heading level0 col25" >25</th>        <th class="col_heading level0 col26" >26</th>        <th class="col_heading level0 col27" >27</th>        <th class="col_heading level0 col28" >28</th>        <th class="col_heading level0 col29" >29</th>        <th class="col_heading level0 col30" >30</th>        <th class="col_heading level0 col31" >31</th>        <th class="col_heading level0 col32" >32</th>        <th class="col_heading level0 col33" >33</th>        <th class="col_heading level0 col34" >34</th>        <th class="col_heading level0 col35" >35</th>        <th class="col_heading level0 col36" >36</th>        <th class="col_heading level0 col37" >37</th>        <th class="col_heading level0 col38" >38</th>        <th class="col_heading level0 col39" >39</th>        <th class="col_heading level0 col40" >40</th>        <th class="col_heading level0 col41" >41</th>        <th class="col_heading level0 col42" >42</th>        <th class="col_heading level0 col43" >43</th>        <th class="col_heading level0 col44" >44</th>        <th class="col_heading level0 col45" >45</th>        <th class="col_heading level0 col46" >46</th>        <th class="col_heading level0 col47" >47</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b7c21_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_b7c21_row0_col0" class="data row0 col0" >6</td>
                        <td id="T_b7c21_row0_col1" class="data row0 col1" >7</td>
                        <td id="T_b7c21_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_b7c21_row0_col3" class="data row0 col3" >1</td>
                        <td id="T_b7c21_row0_col4" class="data row0 col4" >2</td>
                        <td id="T_b7c21_row0_col5" class="data row0 col5" >3</td>
                        <td id="T_b7c21_row0_col6" class="data row0 col6" >4</td>
                        <td id="T_b7c21_row0_col7" class="data row0 col7" >5</td>
                        <td id="T_b7c21_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_b7c21_row0_col9" class="data row0 col9" >17</td>
                        <td id="T_b7c21_row0_col10" class="data row0 col10" >18</td>
                        <td id="T_b7c21_row0_col11" class="data row0 col11" >11</td>
                        <td id="T_b7c21_row0_col12" class="data row0 col12" >12</td>
                        <td id="T_b7c21_row0_col13" class="data row0 col13" >13</td>
                        <td id="T_b7c21_row0_col14" class="data row0 col14" >14</td>
                        <td id="T_b7c21_row0_col15" class="data row0 col15" >15</td>
                        <td id="T_b7c21_row0_col16" class="data row0 col16" >24</td>
                        <td id="T_b7c21_row0_col17" class="data row0 col17" >25</td>
                        <td id="T_b7c21_row0_col18" class="data row0 col18" >26</td>
                        <td id="T_b7c21_row0_col19" class="data row0 col19" >19</td>
                        <td id="T_b7c21_row0_col20" class="data row0 col20" >20</td>
                        <td id="T_b7c21_row0_col21" class="data row0 col21" >21</td>
                        <td id="T_b7c21_row0_col22" class="data row0 col22" >22</td>
                        <td id="T_b7c21_row0_col23" class="data row0 col23" >23</td>
                        <td id="T_b7c21_row0_col24" class="data row0 col24" >32</td>
                        <td id="T_b7c21_row0_col25" class="data row0 col25" >33</td>
                        <td id="T_b7c21_row0_col26" class="data row0 col26" >34</td>
                        <td id="T_b7c21_row0_col27" class="data row0 col27" >27</td>
                        <td id="T_b7c21_row0_col28" class="data row0 col28" >28</td>
                        <td id="T_b7c21_row0_col29" class="data row0 col29" >29</td>
                        <td id="T_b7c21_row0_col30" class="data row0 col30" >30</td>
                        <td id="T_b7c21_row0_col31" class="data row0 col31" >31</td>
                        <td id="T_b7c21_row0_col32" class="data row0 col32" >8</td>
                        <td id="T_b7c21_row0_col33" class="data row0 col33" >9</td>
                        <td id="T_b7c21_row0_col34" class="data row0 col34" >10</td>
                        <td id="T_b7c21_row0_col35" class="data row0 col35" >35</td>
                        <td id="T_b7c21_row0_col36" class="data row0 col36" >36</td>
                        <td id="T_b7c21_row0_col37" class="data row0 col37" >37</td>
                        <td id="T_b7c21_row0_col38" class="data row0 col38" >38</td>
                        <td id="T_b7c21_row0_col39" class="data row0 col39" >39</td>
                        <td id="T_b7c21_row0_col40" class="data row0 col40" >40</td>
                        <td id="T_b7c21_row0_col41" class="data row0 col41" >41</td>
                        <td id="T_b7c21_row0_col42" class="data row0 col42" >42</td>
                        <td id="T_b7c21_row0_col43" class="data row0 col43" >43</td>
                        <td id="T_b7c21_row0_col44" class="data row0 col44" >44</td>
                        <td id="T_b7c21_row0_col45" class="data row0 col45" >45</td>
                        <td id="T_b7c21_row0_col46" class="data row0 col46" >46</td>
                        <td id="T_b7c21_row0_col47" class="data row0 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_b7c21_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_b7c21_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_b7c21_row1_col1" class="data row1 col1" >1</td>
                        <td id="T_b7c21_row1_col2" class="data row1 col2" >18</td>
                        <td id="T_b7c21_row1_col3" class="data row1 col3" >19</td>
                        <td id="T_b7c21_row1_col4" class="data row1 col4" >20</td>
                        <td id="T_b7c21_row1_col5" class="data row1 col5" >5</td>
                        <td id="T_b7c21_row1_col6" class="data row1 col6" >6</td>
                        <td id="T_b7c21_row1_col7" class="data row1 col7" >7</td>
                        <td id="T_b7c21_row1_col8" class="data row1 col8" >8</td>
                        <td id="T_b7c21_row1_col9" class="data row1 col9" >9</td>
                        <td id="T_b7c21_row1_col10" class="data row1 col10" >10</td>
                        <td id="T_b7c21_row1_col11" class="data row1 col11" >11</td>
                        <td id="T_b7c21_row1_col12" class="data row1 col12" >12</td>
                        <td id="T_b7c21_row1_col13" class="data row1 col13" >13</td>
                        <td id="T_b7c21_row1_col14" class="data row1 col14" >14</td>
                        <td id="T_b7c21_row1_col15" class="data row1 col15" >15</td>
                        <td id="T_b7c21_row1_col16" class="data row1 col16" >16</td>
                        <td id="T_b7c21_row1_col17" class="data row1 col17" >17</td>
                        <td id="T_b7c21_row1_col18" class="data row1 col18" >42</td>
                        <td id="T_b7c21_row1_col19" class="data row1 col19" >43</td>
                        <td id="T_b7c21_row1_col20" class="data row1 col20" >44</td>
                        <td id="T_b7c21_row1_col21" class="data row1 col21" >21</td>
                        <td id="T_b7c21_row1_col22" class="data row1 col22" >22</td>
                        <td id="T_b7c21_row1_col23" class="data row1 col23" >23</td>
                        <td id="T_b7c21_row1_col24" class="data row1 col24" >30</td>
                        <td id="T_b7c21_row1_col25" class="data row1 col25" >31</td>
                        <td id="T_b7c21_row1_col26" class="data row1 col26" >24</td>
                        <td id="T_b7c21_row1_col27" class="data row1 col27" >25</td>
                        <td id="T_b7c21_row1_col28" class="data row1 col28" >26</td>
                        <td id="T_b7c21_row1_col29" class="data row1 col29" >27</td>
                        <td id="T_b7c21_row1_col30" class="data row1 col30" >28</td>
                        <td id="T_b7c21_row1_col31" class="data row1 col31" >29</td>
                        <td id="T_b7c21_row1_col32" class="data row1 col32" >4</td>
                        <td id="T_b7c21_row1_col33" class="data row1 col33" >33</td>
                        <td id="T_b7c21_row1_col34" class="data row1 col34" >34</td>
                        <td id="T_b7c21_row1_col35" class="data row1 col35" >35</td>
                        <td id="T_b7c21_row1_col36" class="data row1 col36" >36</td>
                        <td id="T_b7c21_row1_col37" class="data row1 col37" >37</td>
                        <td id="T_b7c21_row1_col38" class="data row1 col38" >2</td>
                        <td id="T_b7c21_row1_col39" class="data row1 col39" >3</td>
                        <td id="T_b7c21_row1_col40" class="data row1 col40" >40</td>
                        <td id="T_b7c21_row1_col41" class="data row1 col41" >41</td>
                        <td id="T_b7c21_row1_col42" class="data row1 col42" >38</td>
                        <td id="T_b7c21_row1_col43" class="data row1 col43" >39</td>
                        <td id="T_b7c21_row1_col44" class="data row1 col44" >32</td>
                        <td id="T_b7c21_row1_col45" class="data row1 col45" >45</td>
                        <td id="T_b7c21_row1_col46" class="data row1 col46" >46</td>
                        <td id="T_b7c21_row1_col47" class="data row1 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_b7c21_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_b7c21_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_b7c21_row2_col1" class="data row2 col1" >1</td>
                        <td id="T_b7c21_row2_col2" class="data row2 col2" >2</td>
                        <td id="T_b7c21_row2_col3" class="data row2 col3" >3</td>
                        <td id="T_b7c21_row2_col4" class="data row2 col4" >10</td>
                        <td id="T_b7c21_row2_col5" class="data row2 col5" >11</td>
                        <td id="T_b7c21_row2_col6" class="data row2 col6" >12</td>
                        <td id="T_b7c21_row2_col7" class="data row2 col7" >7</td>
                        <td id="T_b7c21_row2_col8" class="data row2 col8" >8</td>
                        <td id="T_b7c21_row2_col9" class="data row2 col9" >9</td>
                        <td id="T_b7c21_row2_col10" class="data row2 col10" >40</td>
                        <td id="T_b7c21_row2_col11" class="data row2 col11" >41</td>
                        <td id="T_b7c21_row2_col12" class="data row2 col12" >42</td>
                        <td id="T_b7c21_row2_col13" class="data row2 col13" >13</td>
                        <td id="T_b7c21_row2_col14" class="data row2 col14" >14</td>
                        <td id="T_b7c21_row2_col15" class="data row2 col15" >15</td>
                        <td id="T_b7c21_row2_col16" class="data row2 col16" >22</td>
                        <td id="T_b7c21_row2_col17" class="data row2 col17" >23</td>
                        <td id="T_b7c21_row2_col18" class="data row2 col18" >16</td>
                        <td id="T_b7c21_row2_col19" class="data row2 col19" >17</td>
                        <td id="T_b7c21_row2_col20" class="data row2 col20" >18</td>
                        <td id="T_b7c21_row2_col21" class="data row2 col21" >19</td>
                        <td id="T_b7c21_row2_col22" class="data row2 col22" >20</td>
                        <td id="T_b7c21_row2_col23" class="data row2 col23" >21</td>
                        <td id="T_b7c21_row2_col24" class="data row2 col24" >6</td>
                        <td id="T_b7c21_row2_col25" class="data row2 col25" >25</td>
                        <td id="T_b7c21_row2_col26" class="data row2 col26" >26</td>
                        <td id="T_b7c21_row2_col27" class="data row2 col27" >27</td>
                        <td id="T_b7c21_row2_col28" class="data row2 col28" >28</td>
                        <td id="T_b7c21_row2_col29" class="data row2 col29" >29</td>
                        <td id="T_b7c21_row2_col30" class="data row2 col30" >4</td>
                        <td id="T_b7c21_row2_col31" class="data row2 col31" >5</td>
                        <td id="T_b7c21_row2_col32" class="data row2 col32" >32</td>
                        <td id="T_b7c21_row2_col33" class="data row2 col33" >33</td>
                        <td id="T_b7c21_row2_col34" class="data row2 col34" >34</td>
                        <td id="T_b7c21_row2_col35" class="data row2 col35" >35</td>
                        <td id="T_b7c21_row2_col36" class="data row2 col36" >36</td>
                        <td id="T_b7c21_row2_col37" class="data row2 col37" >37</td>
                        <td id="T_b7c21_row2_col38" class="data row2 col38" >38</td>
                        <td id="T_b7c21_row2_col39" class="data row2 col39" >39</td>
                        <td id="T_b7c21_row2_col40" class="data row2 col40" >30</td>
                        <td id="T_b7c21_row2_col41" class="data row2 col41" >31</td>
                        <td id="T_b7c21_row2_col42" class="data row2 col42" >24</td>
                        <td id="T_b7c21_row2_col43" class="data row2 col43" >43</td>
                        <td id="T_b7c21_row2_col44" class="data row2 col44" >44</td>
                        <td id="T_b7c21_row2_col45" class="data row2 col45" >45</td>
                        <td id="T_b7c21_row2_col46" class="data row2 col46" >46</td>
                        <td id="T_b7c21_row2_col47" class="data row2 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_b7c21_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_b7c21_row3_col0" class="data row3 col0" >36</td>
                        <td id="T_b7c21_row3_col1" class="data row3 col1" >1</td>
                        <td id="T_b7c21_row3_col2" class="data row3 col2" >2</td>
                        <td id="T_b7c21_row3_col3" class="data row3 col3" >3</td>
                        <td id="T_b7c21_row3_col4" class="data row3 col4" >4</td>
                        <td id="T_b7c21_row3_col5" class="data row3 col5" >5</td>
                        <td id="T_b7c21_row3_col6" class="data row3 col6" >34</td>
                        <td id="T_b7c21_row3_col7" class="data row3 col7" >35</td>
                        <td id="T_b7c21_row3_col8" class="data row3 col8" >14</td>
                        <td id="T_b7c21_row3_col9" class="data row3 col9" >15</td>
                        <td id="T_b7c21_row3_col10" class="data row3 col10" >8</td>
                        <td id="T_b7c21_row3_col11" class="data row3 col11" >9</td>
                        <td id="T_b7c21_row3_col12" class="data row3 col12" >10</td>
                        <td id="T_b7c21_row3_col13" class="data row3 col13" >11</td>
                        <td id="T_b7c21_row3_col14" class="data row3 col14" >12</td>
                        <td id="T_b7c21_row3_col15" class="data row3 col15" >13</td>
                        <td id="T_b7c21_row3_col16" class="data row3 col16" >0</td>
                        <td id="T_b7c21_row3_col17" class="data row3 col17" >17</td>
                        <td id="T_b7c21_row3_col18" class="data row3 col18" >18</td>
                        <td id="T_b7c21_row3_col19" class="data row3 col19" >19</td>
                        <td id="T_b7c21_row3_col20" class="data row3 col20" >20</td>
                        <td id="T_b7c21_row3_col21" class="data row3 col21" >21</td>
                        <td id="T_b7c21_row3_col22" class="data row3 col22" >6</td>
                        <td id="T_b7c21_row3_col23" class="data row3 col23" >7</td>
                        <td id="T_b7c21_row3_col24" class="data row3 col24" >24</td>
                        <td id="T_b7c21_row3_col25" class="data row3 col25" >25</td>
                        <td id="T_b7c21_row3_col26" class="data row3 col26" >26</td>
                        <td id="T_b7c21_row3_col27" class="data row3 col27" >27</td>
                        <td id="T_b7c21_row3_col28" class="data row3 col28" >28</td>
                        <td id="T_b7c21_row3_col29" class="data row3 col29" >29</td>
                        <td id="T_b7c21_row3_col30" class="data row3 col30" >30</td>
                        <td id="T_b7c21_row3_col31" class="data row3 col31" >31</td>
                        <td id="T_b7c21_row3_col32" class="data row3 col32" >32</td>
                        <td id="T_b7c21_row3_col33" class="data row3 col33" >33</td>
                        <td id="T_b7c21_row3_col34" class="data row3 col34" >46</td>
                        <td id="T_b7c21_row3_col35" class="data row3 col35" >47</td>
                        <td id="T_b7c21_row3_col36" class="data row3 col36" >40</td>
                        <td id="T_b7c21_row3_col37" class="data row3 col37" >37</td>
                        <td id="T_b7c21_row3_col38" class="data row3 col38" >38</td>
                        <td id="T_b7c21_row3_col39" class="data row3 col39" >39</td>
                        <td id="T_b7c21_row3_col40" class="data row3 col40" >16</td>
                        <td id="T_b7c21_row3_col41" class="data row3 col41" >41</td>
                        <td id="T_b7c21_row3_col42" class="data row3 col42" >42</td>
                        <td id="T_b7c21_row3_col43" class="data row3 col43" >43</td>
                        <td id="T_b7c21_row3_col44" class="data row3 col44" >44</td>
                        <td id="T_b7c21_row3_col45" class="data row3 col45" >45</td>
                        <td id="T_b7c21_row3_col46" class="data row3 col46" >22</td>
                        <td id="T_b7c21_row3_col47" class="data row3 col47" >23</td>
            </tr>
            <tr>
                        <th id="T_b7c21_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_b7c21_row4_col0" class="data row4 col0" >26</td>
                        <td id="T_b7c21_row4_col1" class="data row4 col1" >27</td>
                        <td id="T_b7c21_row4_col2" class="data row4 col2" >28</td>
                        <td id="T_b7c21_row4_col3" class="data row4 col3" >3</td>
                        <td id="T_b7c21_row4_col4" class="data row4 col4" >4</td>
                        <td id="T_b7c21_row4_col5" class="data row4 col5" >5</td>
                        <td id="T_b7c21_row4_col6" class="data row4 col6" >6</td>
                        <td id="T_b7c21_row4_col7" class="data row4 col7" >7</td>
                        <td id="T_b7c21_row4_col8" class="data row4 col8" >2</td>
                        <td id="T_b7c21_row4_col9" class="data row4 col9" >9</td>
                        <td id="T_b7c21_row4_col10" class="data row4 col10" >10</td>
                        <td id="T_b7c21_row4_col11" class="data row4 col11" >11</td>
                        <td id="T_b7c21_row4_col12" class="data row4 col12" >12</td>
                        <td id="T_b7c21_row4_col13" class="data row4 col13" >13</td>
                        <td id="T_b7c21_row4_col14" class="data row4 col14" >0</td>
                        <td id="T_b7c21_row4_col15" class="data row4 col15" >1</td>
                        <td id="T_b7c21_row4_col16" class="data row4 col16" >16</td>
                        <td id="T_b7c21_row4_col17" class="data row4 col17" >17</td>
                        <td id="T_b7c21_row4_col18" class="data row4 col18" >18</td>
                        <td id="T_b7c21_row4_col19" class="data row4 col19" >19</td>
                        <td id="T_b7c21_row4_col20" class="data row4 col20" >20</td>
                        <td id="T_b7c21_row4_col21" class="data row4 col21" >21</td>
                        <td id="T_b7c21_row4_col22" class="data row4 col22" >22</td>
                        <td id="T_b7c21_row4_col23" class="data row4 col23" >23</td>
                        <td id="T_b7c21_row4_col24" class="data row4 col24" >24</td>
                        <td id="T_b7c21_row4_col25" class="data row4 col25" >25</td>
                        <td id="T_b7c21_row4_col26" class="data row4 col26" >44</td>
                        <td id="T_b7c21_row4_col27" class="data row4 col27" >45</td>
                        <td id="T_b7c21_row4_col28" class="data row4 col28" >46</td>
                        <td id="T_b7c21_row4_col29" class="data row4 col29" >29</td>
                        <td id="T_b7c21_row4_col30" class="data row4 col30" >30</td>
                        <td id="T_b7c21_row4_col31" class="data row4 col31" >31</td>
                        <td id="T_b7c21_row4_col32" class="data row4 col32" >38</td>
                        <td id="T_b7c21_row4_col33" class="data row4 col33" >39</td>
                        <td id="T_b7c21_row4_col34" class="data row4 col34" >32</td>
                        <td id="T_b7c21_row4_col35" class="data row4 col35" >33</td>
                        <td id="T_b7c21_row4_col36" class="data row4 col36" >34</td>
                        <td id="T_b7c21_row4_col37" class="data row4 col37" >35</td>
                        <td id="T_b7c21_row4_col38" class="data row4 col38" >36</td>
                        <td id="T_b7c21_row4_col39" class="data row4 col39" >37</td>
                        <td id="T_b7c21_row4_col40" class="data row4 col40" >40</td>
                        <td id="T_b7c21_row4_col41" class="data row4 col41" >41</td>
                        <td id="T_b7c21_row4_col42" class="data row4 col42" >42</td>
                        <td id="T_b7c21_row4_col43" class="data row4 col43" >43</td>
                        <td id="T_b7c21_row4_col44" class="data row4 col44" >14</td>
                        <td id="T_b7c21_row4_col45" class="data row4 col45" >15</td>
                        <td id="T_b7c21_row4_col46" class="data row4 col46" >8</td>
                        <td id="T_b7c21_row4_col47" class="data row4 col47" >47</td>
            </tr>
            <tr>
                        <th id="T_b7c21_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_b7c21_row5_col0" class="data row5 col0" >0</td>
                        <td id="T_b7c21_row5_col1" class="data row5 col1" >1</td>
                        <td id="T_b7c21_row5_col2" class="data row5 col2" >2</td>
                        <td id="T_b7c21_row5_col3" class="data row5 col3" >3</td>
                        <td id="T_b7c21_row5_col4" class="data row5 col4" >4</td>
                        <td id="T_b7c21_row5_col5" class="data row5 col5" >5</td>
                        <td id="T_b7c21_row5_col6" class="data row5 col6" >6</td>
                        <td id="T_b7c21_row5_col7" class="data row5 col7" >7</td>
                        <td id="T_b7c21_row5_col8" class="data row5 col8" >8</td>
                        <td id="T_b7c21_row5_col9" class="data row5 col9" >9</td>
                        <td id="T_b7c21_row5_col10" class="data row5 col10" >10</td>
                        <td id="T_b7c21_row5_col11" class="data row5 col11" >11</td>
                        <td id="T_b7c21_row5_col12" class="data row5 col12" >36</td>
                        <td id="T_b7c21_row5_col13" class="data row5 col13" >37</td>
                        <td id="T_b7c21_row5_col14" class="data row5 col14" >38</td>
                        <td id="T_b7c21_row5_col15" class="data row5 col15" >15</td>
                        <td id="T_b7c21_row5_col16" class="data row5 col16" >16</td>
                        <td id="T_b7c21_row5_col17" class="data row5 col17" >17</td>
                        <td id="T_b7c21_row5_col18" class="data row5 col18" >18</td>
                        <td id="T_b7c21_row5_col19" class="data row5 col19" >19</td>
                        <td id="T_b7c21_row5_col20" class="data row5 col20" >12</td>
                        <td id="T_b7c21_row5_col21" class="data row5 col21" >13</td>
                        <td id="T_b7c21_row5_col22" class="data row5 col22" >14</td>
                        <td id="T_b7c21_row5_col23" class="data row5 col23" >23</td>
                        <td id="T_b7c21_row5_col24" class="data row5 col24" >24</td>
                        <td id="T_b7c21_row5_col25" class="data row5 col25" >25</td>
                        <td id="T_b7c21_row5_col26" class="data row5 col26" >26</td>
                        <td id="T_b7c21_row5_col27" class="data row5 col27" >27</td>
                        <td id="T_b7c21_row5_col28" class="data row5 col28" >20</td>
                        <td id="T_b7c21_row5_col29" class="data row5 col29" >21</td>
                        <td id="T_b7c21_row5_col30" class="data row5 col30" >22</td>
                        <td id="T_b7c21_row5_col31" class="data row5 col31" >31</td>
                        <td id="T_b7c21_row5_col32" class="data row5 col32" >32</td>
                        <td id="T_b7c21_row5_col33" class="data row5 col33" >33</td>
                        <td id="T_b7c21_row5_col34" class="data row5 col34" >34</td>
                        <td id="T_b7c21_row5_col35" class="data row5 col35" >35</td>
                        <td id="T_b7c21_row5_col36" class="data row5 col36" >28</td>
                        <td id="T_b7c21_row5_col37" class="data row5 col37" >29</td>
                        <td id="T_b7c21_row5_col38" class="data row5 col38" >30</td>
                        <td id="T_b7c21_row5_col39" class="data row5 col39" >39</td>
                        <td id="T_b7c21_row5_col40" class="data row5 col40" >46</td>
                        <td id="T_b7c21_row5_col41" class="data row5 col41" >47</td>
                        <td id="T_b7c21_row5_col42" class="data row5 col42" >40</td>
                        <td id="T_b7c21_row5_col43" class="data row5 col43" >41</td>
                        <td id="T_b7c21_row5_col44" class="data row5 col44" >42</td>
                        <td id="T_b7c21_row5_col45" class="data row5 col45" >43</td>
                        <td id="T_b7c21_row5_col46" class="data row5 col46" >44</td>
                        <td id="T_b7c21_row5_col47" class="data row5 col47" >45</td>
            </tr>
    </tbody></table>

Hopefully you should now have some idea what this puzzle is about.
If not, try to stare at these colors for a bit, focusing on each separate row.

_Hint: something about cubes..._

```python
# update the index with somewhat random letters.
MOVES = ["U", "R", "F", "L", "B", "D"]
PERMUTATIONS_DF.index = MOVES

def applyborder(row):
    left_borders = [f'border-left: 5px solid white' if i % 3 == 0  else ''  for i in range(len(row))]
    right_borders = [f'border-right: 5px solid white' if i % 3 == 2  else ''  for i in range(len(row))]
    return [";".join(x) for x in zip(left_borders, right_borders)]

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

def display_row_as_cubes(row, title):
    faces = np.zeros((6,3,3), dtype=np.int8)

    # set the center block in each 3x3 face
    for i in range(6):
        faces[i, 1, 1] = i*8

    # map the value to a 3x3 postition in the right face
    for i, v in enumerate(row):
        color = v
        pos = i % 8
        c = i //8
        x, y = CUBE_MAP[pos]
        faces[c,x, y] = color

    display(pd.DataFrame(np.hstack(faces)).style\
        .hide_index()\
        .set_table_styles([
            {'selector': 'thead', 'props': [('display', 'none')]},
        ])\
        .set_properties(**{
            'font-size': '1pt',
            'width': '30px',
            'height': '30px',
            'color': 'transparent',
            'border': '1px solid darkgrey',
            'text-align': 'center',
            })
        .set_caption(title)\
        .applymap(color_group)\
        .apply(applyborder,axis=1)\
        )

for move in MOVES:
    display_row_as_cubes(PERMUTATIONS_DF.loc[move],f'Move {move}')
```

<style  type="text/css" >
    #T_0cb46_ thead {
          display: none;
    }#T_0cb46_row0_col0,#T_0cb46_row1_col0,#T_0cb46_row2_col0{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_0cb46_row0_col1,#T_0cb46_row1_col1,#T_0cb46_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_0cb46_row0_col2,#T_0cb46_row1_col2,#T_0cb46_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_0cb46_row0_col3,#T_0cb46_row1_col6,#T_0cb46_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_0cb46_row0_col4,#T_0cb46_row1_col7,#T_0cb46_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_0cb46_row0_col5,#T_0cb46_row1_col8,#T_0cb46_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_0cb46_row0_col6,#T_0cb46_row1_col9,#T_0cb46_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_0cb46_row0_col7,#T_0cb46_row1_col10,#T_0cb46_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_0cb46_row0_col8,#T_0cb46_row1_col11,#T_0cb46_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_0cb46_row0_col9,#T_0cb46_row1_col12,#T_0cb46_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_0cb46_row0_col10,#T_0cb46_row1_col13,#T_0cb46_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_0cb46_row0_col11,#T_0cb46_row1_col14,#T_0cb46_row2_col14{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_0cb46_row0_col12,#T_0cb46_row1_col3,#T_0cb46_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_0cb46_row0_col13,#T_0cb46_row1_col4,#T_0cb46_row2_col4{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_0cb46_row0_col14,#T_0cb46_row1_col5,#T_0cb46_row2_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_0cb46_row0_col15,#T_0cb46_row1_col15,#T_0cb46_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_0cb46_row0_col16,#T_0cb46_row1_col16,#T_0cb46_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }#T_0cb46_row0_col17,#T_0cb46_row1_col17,#T_0cb46_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }</style>
<table id="T_0cb46_" ><caption>Move U</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_0cb46_row0_col0" class="data row0 col0" >6</td>
                        <td id="T_0cb46_row0_col1" class="data row0 col1" >7</td>
                        <td id="T_0cb46_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_0cb46_row0_col3" class="data row0 col3" >16</td>
                        <td id="T_0cb46_row0_col4" class="data row0 col4" >17</td>
                        <td id="T_0cb46_row0_col5" class="data row0 col5" >18</td>
                        <td id="T_0cb46_row0_col6" class="data row0 col6" >24</td>
                        <td id="T_0cb46_row0_col7" class="data row0 col7" >25</td>
                        <td id="T_0cb46_row0_col8" class="data row0 col8" >26</td>
                        <td id="T_0cb46_row0_col9" class="data row0 col9" >32</td>
                        <td id="T_0cb46_row0_col10" class="data row0 col10" >33</td>
                        <td id="T_0cb46_row0_col11" class="data row0 col11" >34</td>
                        <td id="T_0cb46_row0_col12" class="data row0 col12" >8</td>
                        <td id="T_0cb46_row0_col13" class="data row0 col13" >9</td>
                        <td id="T_0cb46_row0_col14" class="data row0 col14" >10</td>
                        <td id="T_0cb46_row0_col15" class="data row0 col15" >40</td>
                        <td id="T_0cb46_row0_col16" class="data row0 col16" >41</td>
                        <td id="T_0cb46_row0_col17" class="data row0 col17" >42</td>
            </tr>
            <tr>
                                <td id="T_0cb46_row1_col0" class="data row1 col0" >5</td>
                        <td id="T_0cb46_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_0cb46_row1_col2" class="data row1 col2" >1</td>
                        <td id="T_0cb46_row1_col3" class="data row1 col3" >15</td>
                        <td id="T_0cb46_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_0cb46_row1_col5" class="data row1 col5" >11</td>
                        <td id="T_0cb46_row1_col6" class="data row1 col6" >23</td>
                        <td id="T_0cb46_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_0cb46_row1_col8" class="data row1 col8" >19</td>
                        <td id="T_0cb46_row1_col9" class="data row1 col9" >31</td>
                        <td id="T_0cb46_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_0cb46_row1_col11" class="data row1 col11" >27</td>
                        <td id="T_0cb46_row1_col12" class="data row1 col12" >39</td>
                        <td id="T_0cb46_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_0cb46_row1_col14" class="data row1 col14" >35</td>
                        <td id="T_0cb46_row1_col15" class="data row1 col15" >47</td>
                        <td id="T_0cb46_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_0cb46_row1_col17" class="data row1 col17" >43</td>
            </tr>
            <tr>
                                <td id="T_0cb46_row2_col0" class="data row2 col0" >4</td>
                        <td id="T_0cb46_row2_col1" class="data row2 col1" >3</td>
                        <td id="T_0cb46_row2_col2" class="data row2 col2" >2</td>
                        <td id="T_0cb46_row2_col3" class="data row2 col3" >14</td>
                        <td id="T_0cb46_row2_col4" class="data row2 col4" >13</td>
                        <td id="T_0cb46_row2_col5" class="data row2 col5" >12</td>
                        <td id="T_0cb46_row2_col6" class="data row2 col6" >22</td>
                        <td id="T_0cb46_row2_col7" class="data row2 col7" >21</td>
                        <td id="T_0cb46_row2_col8" class="data row2 col8" >20</td>
                        <td id="T_0cb46_row2_col9" class="data row2 col9" >30</td>
                        <td id="T_0cb46_row2_col10" class="data row2 col10" >29</td>
                        <td id="T_0cb46_row2_col11" class="data row2 col11" >28</td>
                        <td id="T_0cb46_row2_col12" class="data row2 col12" >38</td>
                        <td id="T_0cb46_row2_col13" class="data row2 col13" >37</td>
                        <td id="T_0cb46_row2_col14" class="data row2 col14" >36</td>
                        <td id="T_0cb46_row2_col15" class="data row2 col15" >46</td>
                        <td id="T_0cb46_row2_col16" class="data row2 col16" >45</td>
                        <td id="T_0cb46_row2_col17" class="data row2 col17" >44</td>
            </tr>
    </tbody></table>

<style  type="text/css" >
    #T_bc6fa_ thead {
          display: none;
    }#T_bc6fa_row0_col0,#T_bc6fa_row0_col12,#T_bc6fa_row1_col0,#T_bc6fa_row1_col12,#T_bc6fa_row2_col0,#T_bc6fa_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_bc6fa_row0_col1,#T_bc6fa_row1_col1,#T_bc6fa_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_bc6fa_row0_col2,#T_bc6fa_row1_col2,#T_bc6fa_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_bc6fa_row0_col3,#T_bc6fa_row1_col3,#T_bc6fa_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_bc6fa_row0_col4,#T_bc6fa_row1_col4,#T_bc6fa_row2_col4{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_bc6fa_row0_col5,#T_bc6fa_row1_col5,#T_bc6fa_row2_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_bc6fa_row0_col6,#T_bc6fa_row1_col6,#T_bc6fa_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_bc6fa_row0_col7,#T_bc6fa_row1_col7,#T_bc6fa_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_bc6fa_row0_col8,#T_bc6fa_row1_col8,#T_bc6fa_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }#T_bc6fa_row0_col9,#T_bc6fa_row1_col9,#T_bc6fa_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_bc6fa_row0_col10,#T_bc6fa_row1_col10,#T_bc6fa_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_bc6fa_row0_col11,#T_bc6fa_row1_col11,#T_bc6fa_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_bc6fa_row0_col13,#T_bc6fa_row1_col13,#T_bc6fa_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_bc6fa_row0_col14,#T_bc6fa_row0_col17,#T_bc6fa_row1_col14,#T_bc6fa_row1_col17,#T_bc6fa_row2_col14,#T_bc6fa_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_bc6fa_row0_col15,#T_bc6fa_row1_col15,#T_bc6fa_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_bc6fa_row0_col16,#T_bc6fa_row1_col16,#T_bc6fa_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }</style>
<table id="T_bc6fa_" ><caption>Move R</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_bc6fa_row0_col0" class="data row0 col0" >0</td>
                        <td id="T_bc6fa_row0_col1" class="data row0 col1" >1</td>
                        <td id="T_bc6fa_row0_col2" class="data row0 col2" >18</td>
                        <td id="T_bc6fa_row0_col3" class="data row0 col3" >8</td>
                        <td id="T_bc6fa_row0_col4" class="data row0 col4" >9</td>
                        <td id="T_bc6fa_row0_col5" class="data row0 col5" >10</td>
                        <td id="T_bc6fa_row0_col6" class="data row0 col6" >16</td>
                        <td id="T_bc6fa_row0_col7" class="data row0 col7" >17</td>
                        <td id="T_bc6fa_row0_col8" class="data row0 col8" >42</td>
                        <td id="T_bc6fa_row0_col9" class="data row0 col9" >30</td>
                        <td id="T_bc6fa_row0_col10" class="data row0 col10" >31</td>
                        <td id="T_bc6fa_row0_col11" class="data row0 col11" >24</td>
                        <td id="T_bc6fa_row0_col12" class="data row0 col12" >4</td>
                        <td id="T_bc6fa_row0_col13" class="data row0 col13" >33</td>
                        <td id="T_bc6fa_row0_col14" class="data row0 col14" >34</td>
                        <td id="T_bc6fa_row0_col15" class="data row0 col15" >40</td>
                        <td id="T_bc6fa_row0_col16" class="data row0 col16" >41</td>
                        <td id="T_bc6fa_row0_col17" class="data row0 col17" >38</td>
            </tr>
            <tr>
                                <td id="T_bc6fa_row1_col0" class="data row1 col0" >7</td>
                        <td id="T_bc6fa_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_bc6fa_row1_col2" class="data row1 col2" >19</td>
                        <td id="T_bc6fa_row1_col3" class="data row1 col3" >15</td>
                        <td id="T_bc6fa_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_bc6fa_row1_col5" class="data row1 col5" >11</td>
                        <td id="T_bc6fa_row1_col6" class="data row1 col6" >23</td>
                        <td id="T_bc6fa_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_bc6fa_row1_col8" class="data row1 col8" >43</td>
                        <td id="T_bc6fa_row1_col9" class="data row1 col9" >29</td>
                        <td id="T_bc6fa_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_bc6fa_row1_col11" class="data row1 col11" >25</td>
                        <td id="T_bc6fa_row1_col12" class="data row1 col12" >3</td>
                        <td id="T_bc6fa_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_bc6fa_row1_col14" class="data row1 col14" >35</td>
                        <td id="T_bc6fa_row1_col15" class="data row1 col15" >47</td>
                        <td id="T_bc6fa_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_bc6fa_row1_col17" class="data row1 col17" >39</td>
            </tr>
            <tr>
                                <td id="T_bc6fa_row2_col0" class="data row2 col0" >6</td>
                        <td id="T_bc6fa_row2_col1" class="data row2 col1" >5</td>
                        <td id="T_bc6fa_row2_col2" class="data row2 col2" >20</td>
                        <td id="T_bc6fa_row2_col3" class="data row2 col3" >14</td>
                        <td id="T_bc6fa_row2_col4" class="data row2 col4" >13</td>
                        <td id="T_bc6fa_row2_col5" class="data row2 col5" >12</td>
                        <td id="T_bc6fa_row2_col6" class="data row2 col6" >22</td>
                        <td id="T_bc6fa_row2_col7" class="data row2 col7" >21</td>
                        <td id="T_bc6fa_row2_col8" class="data row2 col8" >44</td>
                        <td id="T_bc6fa_row2_col9" class="data row2 col9" >28</td>
                        <td id="T_bc6fa_row2_col10" class="data row2 col10" >27</td>
                        <td id="T_bc6fa_row2_col11" class="data row2 col11" >26</td>
                        <td id="T_bc6fa_row2_col12" class="data row2 col12" >2</td>
                        <td id="T_bc6fa_row2_col13" class="data row2 col13" >37</td>
                        <td id="T_bc6fa_row2_col14" class="data row2 col14" >36</td>
                        <td id="T_bc6fa_row2_col15" class="data row2 col15" >46</td>
                        <td id="T_bc6fa_row2_col16" class="data row2 col16" >45</td>
                        <td id="T_bc6fa_row2_col17" class="data row2 col17" >32</td>
            </tr>
    </tbody></table>

<style  type="text/css" >
    #T_40eb3_ thead {
          display: none;
    }#T_40eb3_row0_col0,#T_40eb3_row0_col9,#T_40eb3_row1_col0,#T_40eb3_row1_col9,#T_40eb3_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_40eb3_row0_col1,#T_40eb3_row1_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_40eb3_row0_col2,#T_40eb3_row1_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_40eb3_row0_col3,#T_40eb3_row1_col3,#T_40eb3_row2_col0,#T_40eb3_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_40eb3_row0_col4,#T_40eb3_row1_col4,#T_40eb3_row2_col1,#T_40eb3_row2_col4{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_40eb3_row0_col5,#T_40eb3_row1_col5,#T_40eb3_row1_col17,#T_40eb3_row2_col5,#T_40eb3_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }#T_40eb3_row0_col6,#T_40eb3_row1_col6,#T_40eb3_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_40eb3_row0_col7,#T_40eb3_row1_col7,#T_40eb3_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_40eb3_row0_col8,#T_40eb3_row1_col8,#T_40eb3_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_40eb3_row0_col10,#T_40eb3_row0_col16,#T_40eb3_row1_col10,#T_40eb3_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_40eb3_row0_col11,#T_40eb3_row0_col17,#T_40eb3_row1_col11,#T_40eb3_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_40eb3_row0_col12,#T_40eb3_row1_col12,#T_40eb3_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_40eb3_row0_col13,#T_40eb3_row1_col13,#T_40eb3_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_40eb3_row0_col14,#T_40eb3_row1_col14,#T_40eb3_row2_col14{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_40eb3_row0_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_40eb3_row1_col15,#T_40eb3_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_40eb3_row1_col16,#T_40eb3_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }#T_40eb3_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }</style>
<table id="T_40eb3_" ><caption>Move F</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_40eb3_row0_col0" class="data row0 col0" >0</td>
                        <td id="T_40eb3_row0_col1" class="data row0 col1" >1</td>
                        <td id="T_40eb3_row0_col2" class="data row0 col2" >2</td>
                        <td id="T_40eb3_row0_col3" class="data row0 col3" >8</td>
                        <td id="T_40eb3_row0_col4" class="data row0 col4" >9</td>
                        <td id="T_40eb3_row0_col5" class="data row0 col5" >40</td>
                        <td id="T_40eb3_row0_col6" class="data row0 col6" >22</td>
                        <td id="T_40eb3_row0_col7" class="data row0 col7" >23</td>
                        <td id="T_40eb3_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_40eb3_row0_col9" class="data row0 col9" >6</td>
                        <td id="T_40eb3_row0_col10" class="data row0 col10" >25</td>
                        <td id="T_40eb3_row0_col11" class="data row0 col11" >26</td>
                        <td id="T_40eb3_row0_col12" class="data row0 col12" >32</td>
                        <td id="T_40eb3_row0_col13" class="data row0 col13" >33</td>
                        <td id="T_40eb3_row0_col14" class="data row0 col14" >34</td>
                        <td id="T_40eb3_row0_col15" class="data row0 col15" >30</td>
                        <td id="T_40eb3_row0_col16" class="data row0 col16" >31</td>
                        <td id="T_40eb3_row0_col17" class="data row0 col17" >24</td>
            </tr>
            <tr>
                                <td id="T_40eb3_row1_col0" class="data row1 col0" >7</td>
                        <td id="T_40eb3_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_40eb3_row1_col2" class="data row1 col2" >3</td>
                        <td id="T_40eb3_row1_col3" class="data row1 col3" >15</td>
                        <td id="T_40eb3_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_40eb3_row1_col5" class="data row1 col5" >41</td>
                        <td id="T_40eb3_row1_col6" class="data row1 col6" >21</td>
                        <td id="T_40eb3_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_40eb3_row1_col8" class="data row1 col8" >17</td>
                        <td id="T_40eb3_row1_col9" class="data row1 col9" >5</td>
                        <td id="T_40eb3_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_40eb3_row1_col11" class="data row1 col11" >27</td>
                        <td id="T_40eb3_row1_col12" class="data row1 col12" >39</td>
                        <td id="T_40eb3_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_40eb3_row1_col14" class="data row1 col14" >35</td>
                        <td id="T_40eb3_row1_col15" class="data row1 col15" >47</td>
                        <td id="T_40eb3_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_40eb3_row1_col17" class="data row1 col17" >43</td>
            </tr>
            <tr>
                                <td id="T_40eb3_row2_col0" class="data row2 col0" >12</td>
                        <td id="T_40eb3_row2_col1" class="data row2 col1" >11</td>
                        <td id="T_40eb3_row2_col2" class="data row2 col2" >10</td>
                        <td id="T_40eb3_row2_col3" class="data row2 col3" >14</td>
                        <td id="T_40eb3_row2_col4" class="data row2 col4" >13</td>
                        <td id="T_40eb3_row2_col5" class="data row2 col5" >42</td>
                        <td id="T_40eb3_row2_col6" class="data row2 col6" >20</td>
                        <td id="T_40eb3_row2_col7" class="data row2 col7" >19</td>
                        <td id="T_40eb3_row2_col8" class="data row2 col8" >18</td>
                        <td id="T_40eb3_row2_col9" class="data row2 col9" >4</td>
                        <td id="T_40eb3_row2_col10" class="data row2 col10" >29</td>
                        <td id="T_40eb3_row2_col11" class="data row2 col11" >28</td>
                        <td id="T_40eb3_row2_col12" class="data row2 col12" >38</td>
                        <td id="T_40eb3_row2_col13" class="data row2 col13" >37</td>
                        <td id="T_40eb3_row2_col14" class="data row2 col14" >36</td>
                        <td id="T_40eb3_row2_col15" class="data row2 col15" >46</td>
                        <td id="T_40eb3_row2_col16" class="data row2 col16" >45</td>
                        <td id="T_40eb3_row2_col17" class="data row2 col17" >44</td>
            </tr>
    </tbody></table>

<style  type="text/css" >
    #T_58780_ thead {
          display: none;
    }#T_58780_row0_col0,#T_58780_row0_col12,#T_58780_row1_col0,#T_58780_row1_col12,#T_58780_row2_col0,#T_58780_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_58780_row0_col1,#T_58780_row1_col1,#T_58780_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_58780_row0_col2,#T_58780_row1_col2,#T_58780_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_58780_row0_col3,#T_58780_row1_col3,#T_58780_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_58780_row0_col4,#T_58780_row1_col4,#T_58780_row2_col4{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_58780_row0_col5,#T_58780_row1_col5,#T_58780_row2_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_58780_row0_col6,#T_58780_row1_col6,#T_58780_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_58780_row0_col7,#T_58780_row1_col7,#T_58780_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_58780_row0_col8,#T_58780_row1_col8,#T_58780_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_58780_row0_col9,#T_58780_row1_col9,#T_58780_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_58780_row0_col10,#T_58780_row1_col10,#T_58780_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_58780_row0_col11,#T_58780_row1_col11,#T_58780_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_58780_row0_col13,#T_58780_row1_col13,#T_58780_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_58780_row0_col14,#T_58780_row0_col17,#T_58780_row1_col14,#T_58780_row1_col17,#T_58780_row2_col14,#T_58780_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }#T_58780_row0_col15,#T_58780_row1_col15,#T_58780_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_58780_row0_col16,#T_58780_row1_col16,#T_58780_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }</style>
<table id="T_58780_" ><caption>Move L</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_58780_row0_col0" class="data row0 col0" >36</td>
                        <td id="T_58780_row0_col1" class="data row0 col1" >1</td>
                        <td id="T_58780_row0_col2" class="data row0 col2" >2</td>
                        <td id="T_58780_row0_col3" class="data row0 col3" >14</td>
                        <td id="T_58780_row0_col4" class="data row0 col4" >15</td>
                        <td id="T_58780_row0_col5" class="data row0 col5" >8</td>
                        <td id="T_58780_row0_col6" class="data row0 col6" >0</td>
                        <td id="T_58780_row0_col7" class="data row0 col7" >17</td>
                        <td id="T_58780_row0_col8" class="data row0 col8" >18</td>
                        <td id="T_58780_row0_col9" class="data row0 col9" >24</td>
                        <td id="T_58780_row0_col10" class="data row0 col10" >25</td>
                        <td id="T_58780_row0_col11" class="data row0 col11" >26</td>
                        <td id="T_58780_row0_col12" class="data row0 col12" >32</td>
                        <td id="T_58780_row0_col13" class="data row0 col13" >33</td>
                        <td id="T_58780_row0_col14" class="data row0 col14" >46</td>
                        <td id="T_58780_row0_col15" class="data row0 col15" >16</td>
                        <td id="T_58780_row0_col16" class="data row0 col16" >41</td>
                        <td id="T_58780_row0_col17" class="data row0 col17" >42</td>
            </tr>
            <tr>
                                <td id="T_58780_row1_col0" class="data row1 col0" >35</td>
                        <td id="T_58780_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_58780_row1_col2" class="data row1 col2" >3</td>
                        <td id="T_58780_row1_col3" class="data row1 col3" >13</td>
                        <td id="T_58780_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_58780_row1_col5" class="data row1 col5" >9</td>
                        <td id="T_58780_row1_col6" class="data row1 col6" >7</td>
                        <td id="T_58780_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_58780_row1_col8" class="data row1 col8" >19</td>
                        <td id="T_58780_row1_col9" class="data row1 col9" >31</td>
                        <td id="T_58780_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_58780_row1_col11" class="data row1 col11" >27</td>
                        <td id="T_58780_row1_col12" class="data row1 col12" >39</td>
                        <td id="T_58780_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_58780_row1_col14" class="data row1 col14" >47</td>
                        <td id="T_58780_row1_col15" class="data row1 col15" >23</td>
                        <td id="T_58780_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_58780_row1_col17" class="data row1 col17" >43</td>
            </tr>
            <tr>
                                <td id="T_58780_row2_col0" class="data row2 col0" >34</td>
                        <td id="T_58780_row2_col1" class="data row2 col1" >5</td>
                        <td id="T_58780_row2_col2" class="data row2 col2" >4</td>
                        <td id="T_58780_row2_col3" class="data row2 col3" >12</td>
                        <td id="T_58780_row2_col4" class="data row2 col4" >11</td>
                        <td id="T_58780_row2_col5" class="data row2 col5" >10</td>
                        <td id="T_58780_row2_col6" class="data row2 col6" >6</td>
                        <td id="T_58780_row2_col7" class="data row2 col7" >21</td>
                        <td id="T_58780_row2_col8" class="data row2 col8" >20</td>
                        <td id="T_58780_row2_col9" class="data row2 col9" >30</td>
                        <td id="T_58780_row2_col10" class="data row2 col10" >29</td>
                        <td id="T_58780_row2_col11" class="data row2 col11" >28</td>
                        <td id="T_58780_row2_col12" class="data row2 col12" >38</td>
                        <td id="T_58780_row2_col13" class="data row2 col13" >37</td>
                        <td id="T_58780_row2_col14" class="data row2 col14" >40</td>
                        <td id="T_58780_row2_col15" class="data row2 col15" >22</td>
                        <td id="T_58780_row2_col16" class="data row2 col16" >45</td>
                        <td id="T_58780_row2_col17" class="data row2 col17" >44</td>
            </tr>
    </tbody></table>

<style  type="text/css" >
    #T_d9ba5_ thead {
          display: none;
    }#T_d9ba5_row0_col0,#T_d9ba5_row0_col9,#T_d9ba5_row1_col9,#T_d9ba5_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_d9ba5_row0_col1,#T_d9ba5_row0_col10,#T_d9ba5_row1_col10,#T_d9ba5_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_d9ba5_row0_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_d9ba5_row0_col3,#T_d9ba5_row1_col0,#T_d9ba5_row1_col3,#T_d9ba5_row2_col0,#T_d9ba5_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_d9ba5_row0_col4,#T_d9ba5_row1_col4,#T_d9ba5_row2_col4,#T_d9ba5_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_d9ba5_row0_col5,#T_d9ba5_row1_col5,#T_d9ba5_row2_col5,#T_d9ba5_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_d9ba5_row0_col6,#T_d9ba5_row1_col6,#T_d9ba5_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_d9ba5_row0_col7,#T_d9ba5_row1_col7,#T_d9ba5_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_d9ba5_row0_col8,#T_d9ba5_row1_col8,#T_d9ba5_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_d9ba5_row0_col11,#T_d9ba5_row0_col17,#T_d9ba5_row1_col11,#T_d9ba5_row1_col17,#T_d9ba5_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }#T_d9ba5_row0_col12,#T_d9ba5_row1_col12,#T_d9ba5_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_d9ba5_row0_col13,#T_d9ba5_row1_col13,#T_d9ba5_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_d9ba5_row0_col14,#T_d9ba5_row1_col14,#T_d9ba5_row2_col14{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_d9ba5_row0_col15,#T_d9ba5_row1_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_d9ba5_row0_col16,#T_d9ba5_row1_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }#T_d9ba5_row1_col1,#T_d9ba5_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_d9ba5_row1_col2,#T_d9ba5_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_d9ba5_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }</style>
<table id="T_d9ba5_" ><caption>Move B</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_d9ba5_row0_col0" class="data row0 col0" >26</td>
                        <td id="T_d9ba5_row0_col1" class="data row0 col1" >27</td>
                        <td id="T_d9ba5_row0_col2" class="data row0 col2" >28</td>
                        <td id="T_d9ba5_row0_col3" class="data row0 col3" >2</td>
                        <td id="T_d9ba5_row0_col4" class="data row0 col4" >9</td>
                        <td id="T_d9ba5_row0_col5" class="data row0 col5" >10</td>
                        <td id="T_d9ba5_row0_col6" class="data row0 col6" >16</td>
                        <td id="T_d9ba5_row0_col7" class="data row0 col7" >17</td>
                        <td id="T_d9ba5_row0_col8" class="data row0 col8" >18</td>
                        <td id="T_d9ba5_row0_col9" class="data row0 col9" >24</td>
                        <td id="T_d9ba5_row0_col10" class="data row0 col10" >25</td>
                        <td id="T_d9ba5_row0_col11" class="data row0 col11" >44</td>
                        <td id="T_d9ba5_row0_col12" class="data row0 col12" >38</td>
                        <td id="T_d9ba5_row0_col13" class="data row0 col13" >39</td>
                        <td id="T_d9ba5_row0_col14" class="data row0 col14" >32</td>
                        <td id="T_d9ba5_row0_col15" class="data row0 col15" >40</td>
                        <td id="T_d9ba5_row0_col16" class="data row0 col16" >41</td>
                        <td id="T_d9ba5_row0_col17" class="data row0 col17" >42</td>
            </tr>
            <tr>
                                <td id="T_d9ba5_row1_col0" class="data row1 col0" >7</td>
                        <td id="T_d9ba5_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_d9ba5_row1_col2" class="data row1 col2" >3</td>
                        <td id="T_d9ba5_row1_col3" class="data row1 col3" >1</td>
                        <td id="T_d9ba5_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_d9ba5_row1_col5" class="data row1 col5" >11</td>
                        <td id="T_d9ba5_row1_col6" class="data row1 col6" >23</td>
                        <td id="T_d9ba5_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_d9ba5_row1_col8" class="data row1 col8" >19</td>
                        <td id="T_d9ba5_row1_col9" class="data row1 col9" >31</td>
                        <td id="T_d9ba5_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_d9ba5_row1_col11" class="data row1 col11" >45</td>
                        <td id="T_d9ba5_row1_col12" class="data row1 col12" >37</td>
                        <td id="T_d9ba5_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_d9ba5_row1_col14" class="data row1 col14" >33</td>
                        <td id="T_d9ba5_row1_col15" class="data row1 col15" >47</td>
                        <td id="T_d9ba5_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_d9ba5_row1_col17" class="data row1 col17" >43</td>
            </tr>
            <tr>
                                <td id="T_d9ba5_row2_col0" class="data row2 col0" >6</td>
                        <td id="T_d9ba5_row2_col1" class="data row2 col1" >5</td>
                        <td id="T_d9ba5_row2_col2" class="data row2 col2" >4</td>
                        <td id="T_d9ba5_row2_col3" class="data row2 col3" >0</td>
                        <td id="T_d9ba5_row2_col4" class="data row2 col4" >13</td>
                        <td id="T_d9ba5_row2_col5" class="data row2 col5" >12</td>
                        <td id="T_d9ba5_row2_col6" class="data row2 col6" >22</td>
                        <td id="T_d9ba5_row2_col7" class="data row2 col7" >21</td>
                        <td id="T_d9ba5_row2_col8" class="data row2 col8" >20</td>
                        <td id="T_d9ba5_row2_col9" class="data row2 col9" >30</td>
                        <td id="T_d9ba5_row2_col10" class="data row2 col10" >29</td>
                        <td id="T_d9ba5_row2_col11" class="data row2 col11" >46</td>
                        <td id="T_d9ba5_row2_col12" class="data row2 col12" >36</td>
                        <td id="T_d9ba5_row2_col13" class="data row2 col13" >35</td>
                        <td id="T_d9ba5_row2_col14" class="data row2 col14" >34</td>
                        <td id="T_d9ba5_row2_col15" class="data row2 col15" >8</td>
                        <td id="T_d9ba5_row2_col16" class="data row2 col16" >15</td>
                        <td id="T_d9ba5_row2_col17" class="data row2 col17" >14</td>
            </tr>
    </tbody></table>

<style  type="text/css" >
    #T_ab4b7_ thead {
          display: none;
    }#T_ab4b7_row0_col0,#T_ab4b7_row1_col0,#T_ab4b7_row2_col0{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_ab4b7_row0_col1,#T_ab4b7_row1_col1,#T_ab4b7_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_ab4b7_row0_col2,#T_ab4b7_row1_col2,#T_ab4b7_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_ab4b7_row0_col3,#T_ab4b7_row1_col3,#T_ab4b7_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_ab4b7_row0_col4,#T_ab4b7_row1_col4,#T_ab4b7_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_ab4b7_row0_col5,#T_ab4b7_row1_col5,#T_ab4b7_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_ab4b7_row0_col6,#T_ab4b7_row1_col6,#T_ab4b7_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_ab4b7_row0_col7,#T_ab4b7_row1_col7,#T_ab4b7_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_ab4b7_row0_col8,#T_ab4b7_row1_col8,#T_ab4b7_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_ab4b7_row0_col9,#T_ab4b7_row1_col9,#T_ab4b7_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_ab4b7_row0_col10,#T_ab4b7_row1_col10,#T_ab4b7_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_ab4b7_row0_col11,#T_ab4b7_row1_col11,#T_ab4b7_row2_col14{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_ab4b7_row0_col12,#T_ab4b7_row1_col12,#T_ab4b7_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_ab4b7_row0_col13,#T_ab4b7_row1_col13,#T_ab4b7_row2_col4{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_ab4b7_row0_col14,#T_ab4b7_row1_col14,#T_ab4b7_row2_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_ab4b7_row0_col15,#T_ab4b7_row1_col15,#T_ab4b7_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_ab4b7_row0_col16,#T_ab4b7_row1_col16,#T_ab4b7_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }#T_ab4b7_row0_col17,#T_ab4b7_row1_col17,#T_ab4b7_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }</style>
<table id="T_ab4b7_" ><caption>Move D</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_ab4b7_row0_col0" class="data row0 col0" >0</td>
                        <td id="T_ab4b7_row0_col1" class="data row0 col1" >1</td>
                        <td id="T_ab4b7_row0_col2" class="data row0 col2" >2</td>
                        <td id="T_ab4b7_row0_col3" class="data row0 col3" >8</td>
                        <td id="T_ab4b7_row0_col4" class="data row0 col4" >9</td>
                        <td id="T_ab4b7_row0_col5" class="data row0 col5" >10</td>
                        <td id="T_ab4b7_row0_col6" class="data row0 col6" >16</td>
                        <td id="T_ab4b7_row0_col7" class="data row0 col7" >17</td>
                        <td id="T_ab4b7_row0_col8" class="data row0 col8" >18</td>
                        <td id="T_ab4b7_row0_col9" class="data row0 col9" >24</td>
                        <td id="T_ab4b7_row0_col10" class="data row0 col10" >25</td>
                        <td id="T_ab4b7_row0_col11" class="data row0 col11" >26</td>
                        <td id="T_ab4b7_row0_col12" class="data row0 col12" >32</td>
                        <td id="T_ab4b7_row0_col13" class="data row0 col13" >33</td>
                        <td id="T_ab4b7_row0_col14" class="data row0 col14" >34</td>
                        <td id="T_ab4b7_row0_col15" class="data row0 col15" >46</td>
                        <td id="T_ab4b7_row0_col16" class="data row0 col16" >47</td>
                        <td id="T_ab4b7_row0_col17" class="data row0 col17" >40</td>
            </tr>
            <tr>
                                <td id="T_ab4b7_row1_col0" class="data row1 col0" >7</td>
                        <td id="T_ab4b7_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_ab4b7_row1_col2" class="data row1 col2" >3</td>
                        <td id="T_ab4b7_row1_col3" class="data row1 col3" >15</td>
                        <td id="T_ab4b7_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_ab4b7_row1_col5" class="data row1 col5" >11</td>
                        <td id="T_ab4b7_row1_col6" class="data row1 col6" >23</td>
                        <td id="T_ab4b7_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_ab4b7_row1_col8" class="data row1 col8" >19</td>
                        <td id="T_ab4b7_row1_col9" class="data row1 col9" >31</td>
                        <td id="T_ab4b7_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_ab4b7_row1_col11" class="data row1 col11" >27</td>
                        <td id="T_ab4b7_row1_col12" class="data row1 col12" >39</td>
                        <td id="T_ab4b7_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_ab4b7_row1_col14" class="data row1 col14" >35</td>
                        <td id="T_ab4b7_row1_col15" class="data row1 col15" >45</td>
                        <td id="T_ab4b7_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_ab4b7_row1_col17" class="data row1 col17" >41</td>
            </tr>
            <tr>
                                <td id="T_ab4b7_row2_col0" class="data row2 col0" >6</td>
                        <td id="T_ab4b7_row2_col1" class="data row2 col1" >5</td>
                        <td id="T_ab4b7_row2_col2" class="data row2 col2" >4</td>
                        <td id="T_ab4b7_row2_col3" class="data row2 col3" >38</td>
                        <td id="T_ab4b7_row2_col4" class="data row2 col4" >37</td>
                        <td id="T_ab4b7_row2_col5" class="data row2 col5" >36</td>
                        <td id="T_ab4b7_row2_col6" class="data row2 col6" >14</td>
                        <td id="T_ab4b7_row2_col7" class="data row2 col7" >13</td>
                        <td id="T_ab4b7_row2_col8" class="data row2 col8" >12</td>
                        <td id="T_ab4b7_row2_col9" class="data row2 col9" >22</td>
                        <td id="T_ab4b7_row2_col10" class="data row2 col10" >21</td>
                        <td id="T_ab4b7_row2_col11" class="data row2 col11" >20</td>
                        <td id="T_ab4b7_row2_col12" class="data row2 col12" >30</td>
                        <td id="T_ab4b7_row2_col13" class="data row2 col13" >29</td>
                        <td id="T_ab4b7_row2_col14" class="data row2 col14" >28</td>
                        <td id="T_ab4b7_row2_col15" class="data row2 col15" >44</td>
                        <td id="T_ab4b7_row2_col16" class="data row2 col16" >43</td>
                        <td id="T_ab4b7_row2_col17" class="data row2 col17" >42</td>
            </tr>
    </tbody></table>

Hopefully you figured out that these permutations are 6 of the possible moves that you can perform on a **Rubik's Cube**!

This means that the elements of the `sol` array define a sequence of moves that we apply on the initial state that will transform it into the final solved state.

Let's plot both of these states to get a better idea of how we should proceed.

```python
display_row_as_cubes([8*i for i in INITIAL_VALUES], "initial state")
display_row_as_cubes([8*i for i in FINAL_STATE_VALUES], "final state")
```

<style  type="text/css" >
    #T_ebb44_ thead {
          display: none;
    }#T_ebb44_row0_col0,#T_ebb44_row1_col9,#T_ebb44_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_ebb44_row0_col1,#T_ebb44_row1_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_ebb44_row0_col2,#T_ebb44_row0_col5,#T_ebb44_row0_col8,#T_ebb44_row1_col11,#T_ebb44_row1_col17,#T_ebb44_row2_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_ebb44_row0_col3,#T_ebb44_row0_col15,#T_ebb44_row2_col0,#T_ebb44_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_ebb44_row0_col4,#T_ebb44_row1_col4,#T_ebb44_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_ebb44_row0_col6,#T_ebb44_row1_col6,#T_ebb44_row1_col12,#T_ebb44_row1_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_ebb44_row0_col7,#T_ebb44_row0_col13,#T_ebb44_row0_col16,#T_ebb44_row1_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_ebb44_row0_col9,#T_ebb44_row0_col12,#T_ebb44_row1_col3,#T_ebb44_row2_col9,#T_ebb44_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_ebb44_row0_col10,#T_ebb44_row1_col7,#T_ebb44_row2_col10,#T_ebb44_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_ebb44_row0_col11,#T_ebb44_row0_col14,#T_ebb44_row1_col8,#T_ebb44_row1_col14,#T_ebb44_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_ebb44_row0_col17,#T_ebb44_row1_col2,#T_ebb44_row2_col2,#T_ebb44_row2_col14{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_ebb44_row1_col0{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_ebb44_row1_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_ebb44_row1_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_ebb44_row1_col16,#T_ebb44_row2_col4,#T_ebb44_row2_col7,#T_ebb44_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }#T_ebb44_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_ebb44_row2_col8,#T_ebb44_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }</style>
<table id="T_ebb44_" ><caption>initial state</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_ebb44_row0_col0" class="data row0 col0" >40</td>
                        <td id="T_ebb44_row0_col1" class="data row0 col1" >0</td>
                        <td id="T_ebb44_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_ebb44_row0_col3" class="data row0 col3" >32</td>
                        <td id="T_ebb44_row0_col4" class="data row0 col4" >8</td>
                        <td id="T_ebb44_row0_col5" class="data row0 col5" >0</td>
                        <td id="T_ebb44_row0_col6" class="data row0 col6" >24</td>
                        <td id="T_ebb44_row0_col7" class="data row0 col7" >32</td>
                        <td id="T_ebb44_row0_col8" class="data row0 col8" >0</td>
                        <td id="T_ebb44_row0_col9" class="data row0 col9" >16</td>
                        <td id="T_ebb44_row0_col10" class="data row0 col10" >16</td>
                        <td id="T_ebb44_row0_col11" class="data row0 col11" >8</td>
                        <td id="T_ebb44_row0_col12" class="data row0 col12" >16</td>
                        <td id="T_ebb44_row0_col13" class="data row0 col13" >32</td>
                        <td id="T_ebb44_row0_col14" class="data row0 col14" >8</td>
                        <td id="T_ebb44_row0_col15" class="data row0 col15" >32</td>
                        <td id="T_ebb44_row0_col16" class="data row0 col16" >32</td>
                        <td id="T_ebb44_row0_col17" class="data row0 col17" >24</td>
            </tr>
            <tr>
                                <td id="T_ebb44_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_ebb44_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_ebb44_row1_col2" class="data row1 col2" >24</td>
                        <td id="T_ebb44_row1_col3" class="data row1 col3" >16</td>
                        <td id="T_ebb44_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_ebb44_row1_col5" class="data row1 col5" >32</td>
                        <td id="T_ebb44_row1_col6" class="data row1 col6" >24</td>
                        <td id="T_ebb44_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_ebb44_row1_col8" class="data row1 col8" >8</td>
                        <td id="T_ebb44_row1_col9" class="data row1 col9" >40</td>
                        <td id="T_ebb44_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_ebb44_row1_col11" class="data row1 col11" >0</td>
                        <td id="T_ebb44_row1_col12" class="data row1 col12" >24</td>
                        <td id="T_ebb44_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_ebb44_row1_col14" class="data row1 col14" >8</td>
                        <td id="T_ebb44_row1_col15" class="data row1 col15" >24</td>
                        <td id="T_ebb44_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_ebb44_row1_col17" class="data row1 col17" >0</td>
            </tr>
            <tr>
                                <td id="T_ebb44_row2_col0" class="data row2 col0" >32</td>
                        <td id="T_ebb44_row2_col1" class="data row2 col1" >8</td>
                        <td id="T_ebb44_row2_col2" class="data row2 col2" >24</td>
                        <td id="T_ebb44_row2_col3" class="data row2 col3" >32</td>
                        <td id="T_ebb44_row2_col4" class="data row2 col4" >40</td>
                        <td id="T_ebb44_row2_col5" class="data row2 col5" >0</td>
                        <td id="T_ebb44_row2_col6" class="data row2 col6" >8</td>
                        <td id="T_ebb44_row2_col7" class="data row2 col7" >40</td>
                        <td id="T_ebb44_row2_col8" class="data row2 col8" >40</td>
                        <td id="T_ebb44_row2_col9" class="data row2 col9" >16</td>
                        <td id="T_ebb44_row2_col10" class="data row2 col10" >16</td>
                        <td id="T_ebb44_row2_col11" class="data row2 col11" >8</td>
                        <td id="T_ebb44_row2_col12" class="data row2 col12" >16</td>
                        <td id="T_ebb44_row2_col13" class="data row2 col13" >16</td>
                        <td id="T_ebb44_row2_col14" class="data row2 col14" >24</td>
                        <td id="T_ebb44_row2_col15" class="data row2 col15" >40</td>
                        <td id="T_ebb44_row2_col16" class="data row2 col16" >40</td>
                        <td id="T_ebb44_row2_col17" class="data row2 col17" >40</td>
            </tr>
    </tbody></table>

<style  type="text/css" >
    #T_31714_ thead {
          display: none;
    }#T_31714_row0_col0,#T_31714_row1_col0,#T_31714_row2_col0{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_31714_row0_col1,#T_31714_row1_col1,#T_31714_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_31714_row0_col2,#T_31714_row1_col2,#T_31714_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_31714_row0_col3,#T_31714_row1_col3,#T_31714_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_31714_row0_col4,#T_31714_row1_col4,#T_31714_row2_col4{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_31714_row0_col5,#T_31714_row1_col5,#T_31714_row2_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_31714_row0_col6,#T_31714_row1_col6,#T_31714_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_31714_row0_col7,#T_31714_row1_col7,#T_31714_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_31714_row0_col8,#T_31714_row1_col8,#T_31714_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_31714_row0_col9,#T_31714_row1_col9,#T_31714_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_31714_row0_col10,#T_31714_row1_col10,#T_31714_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_31714_row0_col11,#T_31714_row1_col11,#T_31714_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_31714_row0_col12,#T_31714_row1_col12,#T_31714_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_31714_row0_col13,#T_31714_row1_col13,#T_31714_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_31714_row0_col14,#T_31714_row1_col14,#T_31714_row2_col14{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_31714_row0_col15,#T_31714_row1_col15,#T_31714_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_31714_row0_col16,#T_31714_row1_col16,#T_31714_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }#T_31714_row0_col17,#T_31714_row1_col17,#T_31714_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }</style>
<table id="T_31714_" ><caption>final state</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_31714_row0_col0" class="data row0 col0" >0</td>
                        <td id="T_31714_row0_col1" class="data row0 col1" >0</td>
                        <td id="T_31714_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_31714_row0_col3" class="data row0 col3" >8</td>
                        <td id="T_31714_row0_col4" class="data row0 col4" >8</td>
                        <td id="T_31714_row0_col5" class="data row0 col5" >8</td>
                        <td id="T_31714_row0_col6" class="data row0 col6" >16</td>
                        <td id="T_31714_row0_col7" class="data row0 col7" >16</td>
                        <td id="T_31714_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_31714_row0_col9" class="data row0 col9" >24</td>
                        <td id="T_31714_row0_col10" class="data row0 col10" >24</td>
                        <td id="T_31714_row0_col11" class="data row0 col11" >24</td>
                        <td id="T_31714_row0_col12" class="data row0 col12" >32</td>
                        <td id="T_31714_row0_col13" class="data row0 col13" >32</td>
                        <td id="T_31714_row0_col14" class="data row0 col14" >32</td>
                        <td id="T_31714_row0_col15" class="data row0 col15" >40</td>
                        <td id="T_31714_row0_col16" class="data row0 col16" >40</td>
                        <td id="T_31714_row0_col17" class="data row0 col17" >40</td>
            </tr>
            <tr>
                                <td id="T_31714_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_31714_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_31714_row1_col2" class="data row1 col2" >0</td>
                        <td id="T_31714_row1_col3" class="data row1 col3" >8</td>
                        <td id="T_31714_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_31714_row1_col5" class="data row1 col5" >8</td>
                        <td id="T_31714_row1_col6" class="data row1 col6" >16</td>
                        <td id="T_31714_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_31714_row1_col8" class="data row1 col8" >16</td>
                        <td id="T_31714_row1_col9" class="data row1 col9" >24</td>
                        <td id="T_31714_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_31714_row1_col11" class="data row1 col11" >24</td>
                        <td id="T_31714_row1_col12" class="data row1 col12" >32</td>
                        <td id="T_31714_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_31714_row1_col14" class="data row1 col14" >32</td>
                        <td id="T_31714_row1_col15" class="data row1 col15" >40</td>
                        <td id="T_31714_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_31714_row1_col17" class="data row1 col17" >40</td>
            </tr>
            <tr>
                                <td id="T_31714_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_31714_row2_col1" class="data row2 col1" >0</td>
                        <td id="T_31714_row2_col2" class="data row2 col2" >0</td>
                        <td id="T_31714_row2_col3" class="data row2 col3" >8</td>
                        <td id="T_31714_row2_col4" class="data row2 col4" >8</td>
                        <td id="T_31714_row2_col5" class="data row2 col5" >8</td>
                        <td id="T_31714_row2_col6" class="data row2 col6" >16</td>
                        <td id="T_31714_row2_col7" class="data row2 col7" >16</td>
                        <td id="T_31714_row2_col8" class="data row2 col8" >16</td>
                        <td id="T_31714_row2_col9" class="data row2 col9" >24</td>
                        <td id="T_31714_row2_col10" class="data row2 col10" >24</td>
                        <td id="T_31714_row2_col11" class="data row2 col11" >24</td>
                        <td id="T_31714_row2_col12" class="data row2 col12" >32</td>
                        <td id="T_31714_row2_col13" class="data row2 col13" >32</td>
                        <td id="T_31714_row2_col14" class="data row2 col14" >32</td>
                        <td id="T_31714_row2_col15" class="data row2 col15" >40</td>
                        <td id="T_31714_row2_col16" class="data row2 col16" >40</td>
                        <td id="T_31714_row2_col17" class="data row2 col17" >40</td>
            </tr>
    </tbody></table>

We won't write our own solver, since many great ones already exist.
The one we use is [rubiks-cube-solver.com](https://rubiks-cube-solver.com/solution.php?cube=0611114524521325561451432266332641332352452334554461666) which has a nice interface for inputting the faces (the link should take you to a page where the cube is already input).

We get the following 20 move solution from the solver site.
Unfortunately, this includes counter-clockwise and double moves, which we can't perform directly with the given set of moves.

```python
CUBE_SOL_SHORT = [
    "L", "U", "L'", "U2", "F'", "D'", "F2", "B'", "U'", "L'",
    "F2", "U'", "D'", "L2","U", "B2", "U", "B2", "R2", "L2"]
```

This little snippet should convert any *"2"*s and *" ' "*s (counter-clockwise moves) to the appropriate number of repetitions of the move.
We also map the move letter to the original matrix row.

This will give us our final solution vector.

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
print("sol:")
print(SOL)
```

    sol:
    [3, 0, 3, 3, 3, 0, 0, 2, 2, 2, 5, 5, 5, 2, 2, 4, 4, 4, 0, 0, 0, 3, 3, 3, 2, 2, 0, 0, 0, 5, 5, 5, 3, 3, 0, 4, 4, 0, 4, 4, 1, 1, 3, 3]

Finally, we can check that our solution works as intended with our Python translations of the original Cairo functions.

```python
state = { i:v for i,v in enumerate(INITIAL_VALUES)}

run(state, DATA, SOL)

solved_state = [x for _, x in state.items()]
assert solved_state == list(FINAL_STATE_VALUES), "state is not solved"

display_row_as_cubes([8*i for i in solved_state], "solved state")
```

<style  type="text/css" >
    #T_f950a_ thead {
          display: none;
    }    #T_f950a_ table {
          table-layout: fixed;
          text-align: center;
          margin-left: auto;
          margin-right: auto;
    }    #T_f950a_ caption {
          text-align: center;
    }#T_f950a_row0_col0,#T_f950a_row1_col0,#T_f950a_row2_col0{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            border-left:  5px solid white;
        }#T_f950a_row0_col1,#T_f950a_row1_col1,#T_f950a_row2_col1{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
        }#T_f950a_row0_col2,#T_f950a_row1_col2,#T_f950a_row2_col2{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  lightgrey;
            : ;
            border-right:  5px solid white;
        }#T_f950a_row0_col3,#T_f950a_row1_col3,#T_f950a_row2_col3{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            border-left:  5px solid white;
        }#T_f950a_row0_col4,#T_f950a_row1_col4,#T_f950a_row2_col4{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
        }#T_f950a_row0_col5,#T_f950a_row1_col5,#T_f950a_row2_col5{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  orange;
            : ;
            border-right:  5px solid white;
        }#T_f950a_row0_col6,#T_f950a_row1_col6,#T_f950a_row2_col6{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            border-left:  5px solid white;
        }#T_f950a_row0_col7,#T_f950a_row1_col7,#T_f950a_row2_col7{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
        }#T_f950a_row0_col8,#T_f950a_row1_col8,#T_f950a_row2_col8{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  green;
            : ;
            border-right:  5px solid white;
        }#T_f950a_row0_col9,#T_f950a_row1_col9,#T_f950a_row2_col9{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            border-left:  5px solid white;
        }#T_f950a_row0_col10,#T_f950a_row1_col10,#T_f950a_row2_col10{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
        }#T_f950a_row0_col11,#T_f950a_row1_col11,#T_f950a_row2_col11{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  red;
            : ;
            border-right:  5px solid white;
        }#T_f950a_row0_col12,#T_f950a_row1_col12,#T_f950a_row2_col12{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            border-left:  5px solid white;
        }#T_f950a_row0_col13,#T_f950a_row1_col13,#T_f950a_row2_col13{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
        }#T_f950a_row0_col14,#T_f950a_row1_col14,#T_f950a_row2_col14{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  blue;
            : ;
            border-right:  5px solid white;
        }#T_f950a_row0_col15,#T_f950a_row1_col15,#T_f950a_row2_col15{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            border-left:  5px solid white;
        }#T_f950a_row0_col16,#T_f950a_row1_col16,#T_f950a_row2_col16{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
        }#T_f950a_row0_col17,#T_f950a_row1_col17,#T_f950a_row2_col17{
            font-size:  1pt;
            width:  30px;
            height:  30px;
            color:  transparent;
            border:  1px solid darkgrey;
            text-align:  center;
            background-color:  yellow;
            : ;
            border-right:  5px solid white;
        }</style>
<table id="T_f950a_" ><caption>solved state</caption><thead>    <tr>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_f950a_row0_col0" class="data row0 col0" >0</td>
                        <td id="T_f950a_row0_col1" class="data row0 col1" >0</td>
                        <td id="T_f950a_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_f950a_row0_col3" class="data row0 col3" >8</td>
                        <td id="T_f950a_row0_col4" class="data row0 col4" >8</td>
                        <td id="T_f950a_row0_col5" class="data row0 col5" >8</td>
                        <td id="T_f950a_row0_col6" class="data row0 col6" >16</td>
                        <td id="T_f950a_row0_col7" class="data row0 col7" >16</td>
                        <td id="T_f950a_row0_col8" class="data row0 col8" >16</td>
                        <td id="T_f950a_row0_col9" class="data row0 col9" >24</td>
                        <td id="T_f950a_row0_col10" class="data row0 col10" >24</td>
                        <td id="T_f950a_row0_col11" class="data row0 col11" >24</td>
                        <td id="T_f950a_row0_col12" class="data row0 col12" >32</td>
                        <td id="T_f950a_row0_col13" class="data row0 col13" >32</td>
                        <td id="T_f950a_row0_col14" class="data row0 col14" >32</td>
                        <td id="T_f950a_row0_col15" class="data row0 col15" >40</td>
                        <td id="T_f950a_row0_col16" class="data row0 col16" >40</td>
                        <td id="T_f950a_row0_col17" class="data row0 col17" >40</td>
            </tr>
            <tr>
                                <td id="T_f950a_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_f950a_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_f950a_row1_col2" class="data row1 col2" >0</td>
                        <td id="T_f950a_row1_col3" class="data row1 col3" >8</td>
                        <td id="T_f950a_row1_col4" class="data row1 col4" >8</td>
                        <td id="T_f950a_row1_col5" class="data row1 col5" >8</td>
                        <td id="T_f950a_row1_col6" class="data row1 col6" >16</td>
                        <td id="T_f950a_row1_col7" class="data row1 col7" >16</td>
                        <td id="T_f950a_row1_col8" class="data row1 col8" >16</td>
                        <td id="T_f950a_row1_col9" class="data row1 col9" >24</td>
                        <td id="T_f950a_row1_col10" class="data row1 col10" >24</td>
                        <td id="T_f950a_row1_col11" class="data row1 col11" >24</td>
                        <td id="T_f950a_row1_col12" class="data row1 col12" >32</td>
                        <td id="T_f950a_row1_col13" class="data row1 col13" >32</td>
                        <td id="T_f950a_row1_col14" class="data row1 col14" >32</td>
                        <td id="T_f950a_row1_col15" class="data row1 col15" >40</td>
                        <td id="T_f950a_row1_col16" class="data row1 col16" >40</td>
                        <td id="T_f950a_row1_col17" class="data row1 col17" >40</td>
            </tr>
            <tr>
                                <td id="T_f950a_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_f950a_row2_col1" class="data row2 col1" >0</td>
                        <td id="T_f950a_row2_col2" class="data row2 col2" >0</td>
                        <td id="T_f950a_row2_col3" class="data row2 col3" >8</td>
                        <td id="T_f950a_row2_col4" class="data row2 col4" >8</td>
                        <td id="T_f950a_row2_col5" class="data row2 col5" >8</td>
                        <td id="T_f950a_row2_col6" class="data row2 col6" >16</td>
                        <td id="T_f950a_row2_col7" class="data row2 col7" >16</td>
                        <td id="T_f950a_row2_col8" class="data row2 col8" >16</td>
                        <td id="T_f950a_row2_col9" class="data row2 col9" >24</td>
                        <td id="T_f950a_row2_col10" class="data row2 col10" >24</td>
                        <td id="T_f950a_row2_col11" class="data row2 col11" >24</td>
                        <td id="T_f950a_row2_col12" class="data row2 col12" >32</td>
                        <td id="T_f950a_row2_col13" class="data row2 col13" >32</td>
                        <td id="T_f950a_row2_col14" class="data row2 col14" >32</td>
                        <td id="T_f950a_row2_col15" class="data row2 col15" >40</td>
                        <td id="T_f950a_row2_col16" class="data row2 col16" >40</td>
                        <td id="T_f950a_row2_col17" class="data row2 col17" >40</td>
            </tr>
    </tbody></table>

Success!
Our moves can correctly solve the this intial cube.

## 3. Writing the hints

The last, final step is to write the appropriate solutions we found along the way as Cairo hints.

### 3.1 `get_initial_value()`

The hint we provide here is simply the `initial_value` list we found previously.
We need to write the 49 elements (including the initial 48) in the `initial_value` array.

```javascript
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

### 3.2 `main()`

The `run` function expects an array `sol` and its length `sol_size`, which are easy to set using `segements.write_arg`.

We also need to initialize the `state` with all the state updates we performed along the way.
This took some time to figure out, but in the end I resorted to re-running the `run` function and saving the 3-tuple of DictAccess values in a single long array.

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

If you want to try this yourself, head over to the [Cairo Playground](https://cairo-lang.org/playground/) and select the **Amun** puzzle.
Fill in the appropriate hints (and set `0x0BADBEEF` to your mainnet Etherium address) and watch everything compile!

## 4. Conclusion

This puzzle was an incredibly fun challenge to solve, but unfortunately I was just a bit slower that [William Borgeaud](https://solvable.group) who solved it a full 4 hours earlier,
so hats off to you William!

Overall it was a fun to get a better understanding of Cairo, and I'll be looking forward to doing more with it than solve puzzles.
It felt like there's a lot of potential in the language, but it definitely takes some getting used to.
Hopefully STARKWARE will do another round of the Cairo Games soon!

<!-- **P.S**: We're looking for awesome people to work with us at [Taurus](https://www.linkedin.com/company/taurus-group-sa/jobs/) -->
