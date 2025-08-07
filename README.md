# drug_name_generator

This is a deployable dash app that generates [new] drug names, and displays them in a word cloud. 

## Installing Dependencies
To install dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### Data Download and Preprocessing
You can use a direct download link to obtain current brand drug names from the FDA: https://www.fda.gov/media/89850/download?attachment

The app.py file expects a `product.txt` file in the `data` folder.

### Generative Model
To call this a generative model is probably overkill - this is a naive algorithm that uses regexes to extract [plausible] prefix, middle, and suffix tokens from existing brand drug names. The tokens are generated using a few simple regexes: we split on one or two consecutive vowels (with preference for two consecutive vowels), and then recombine single characters at the beginning or the ends to create true prefixes and suffixes. 

Consider the four word drug name "corpus" below.

EXAMPLES: 
(1) wegovy --> we + go + vy
(2) xanax --> xa + na + x --> xa + nax
(3) mounjaro --> mou + nja + ro
(4) amyvid --> a + my + vi + d --> amy + vid (recombining single characters)

We can now extract the following information.

Prefixes: [we, xa, zia, amy]
Middle: [go, nja]
Suffixes: [vy, nd, na, vid]

Then we can calculate the count and relative frequency (probability) of each prefix, middle, and suffix within its set. In this simplified example, each prefix would have a count 1 and a probability of 1/4. To create a new drug name, we select a prefix, middle, and suffix at random based on their probabilities, such as `we + nja + vy = wenjavy`

### Word Cloud Visualization
A silly drug name generator deserves a surprisingly elegant looking app with fun and completely unnecessary features. There are two sliders: (1) the temperature slider determines how selective we are (where a higher temperature means we allow lower probability tokens to be selected), and (2) the number of drug slider allows us to display anywhere from 5-15 newly generate drug names in a cloud. Moving either slider automatically generates new drug names, and clicking the [Regenerate Drug Names] button generates new drug names with the current slider settings. Higher probability drug names are displayed in a larger font.

### Running the app
Using terminal, cd into this directory and run the following command: ```python3 app.py```

### Future Work
Currently there is only one corpus of brand drug names and the data set is rather small with only 6,986 drug names. It would be interesting to expand the size of this data set, and also introduce trade classes for specific types of drugs so that new drugs for a certain trade class can be constructed using prefix + middle + suffix from existing drugs in that aprticualr trade class.