# Interpret-flair

##### Please note: this repository is not officially associated with [Flair](https://github.com/flairNLP/flair) nor with [Captum](https://github.com/pytorch/captum).
This notebook shows an attempt at integrating [Captum](https://github.com/pytorch/captum) with a custom trained [Flair text-classifier](https://github.com/flairNLP/flair).
As such, this approach should also be validated by outsiders.


The example model was trained on the [BBC dataset](https://www.kaggle.com/c/learn-ai-bbc/overview) and makes use of sentence-transformers' [xlm-r-100langs-bert-base-nli-mean-tokens model](https://huggingface.co/sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens).
The model has the following results:
- F-score (micro) 0.964
- F-score (macro) 0.9626
- Accuracy 0.964


```python
import numpy as np
import flair
from flair.models import TextClassifier
from flair.data import Sentence
from interpretation_package.flair_model_wrapper import ModelWrapper
from interpretation_package.interpret_flair import interpret_sentence, visualize_attributions
from captum.attr import LayerIntegratedGradients
```

Define which device to use: 'cpu' or 'cuda'

```python
flair.device = 'cuda'
```

We load the trained Flair classifier.

```python
model_path = "./model/output/best-model.pt"
```

```python
flair_model = TextClassifier.load(model_path)
```

    2020-11-21 20:58:55,379 loading file ./model/output/best-model.pt


In order to make use of Captum's [LayerIntegratedGradients method](https://captum.ai/api/layer.html#layer-integrated-gradients) we had to rework Flair's forward function. This is handled by the wrapper.
The wrapper inherits functions of the Flair [text-classifier object](https://github.com/flairNLP/flair/blob/master/flair/models/text_classification_model.py) and allows us to calculate attributions with respect to a target class.


```python
flair_model_wrapper = ModelWrapper(flair_model)
```

Let's check out the underlying XLMRoberta model.


```python
print(flair_model_wrapper.model)
```

    XLMRobertaModel(
      (embeddings): RobertaEmbeddings(
        (word_embeddings): Embedding(250002, 768, padding_idx=1)
        (position_embeddings): Embedding(514, 768, padding_idx=1)
        (token_type_embeddings): Embedding(1, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): RobertaEncoder(
        (layer): ModuleList(
          (0): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (1): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (2): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (3): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (4): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (5): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (7): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (8): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (9): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (10): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): RobertaPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )


As described in the source code of [documentation of Captum](https://github.com/pytorch/captum/blob/master/captum/attr/_core/layer/layer_integrated_gradients.py):


"*Layer Integrated Gradients is a variant of Integrated Gradients that assigns
an importance score to layer inputs or outputs, depending on whether we
attribute to the former or to the latter one.*"


In this case, we are interested how the input embeddings of the model contribute to the output.


```python
lig = LayerIntegratedGradients(flair_model_wrapper, flair_model_wrapper.model.embeddings)
```

To test this, let's take the two paragraphs of an article about business by the Economist.


[Which Japanese mogul will leave the biggest legacy?](https://www.economist.com/business/2020/11/07/which-japanese-mogul-will-leave-the-biggest-legacy)


```python
sentence = """
In the 1990s, when a youthful Son Masayoshi, a Japanese entrepreneur, was pursuing acquisitions in his home country, he sought advice from a banker eight years his junior called Mikitani Hiroshi. They shared a lot in common: both had studied in America (Mr Son at the University of California, Berkeley, Mr Mikitani at Harvard Business School); they had a common interest in the internet; and they were both baseball mad. In the decades since, both men have blazed past a stifling corporate hierarchy to become two of Japan’s leading tech billionaires. 
Mr Mikitani, who says in an interview that he did not even know the word “entrepreneur” when he enrolled at Harvard, pioneered e-commerce in Japan via Rakuten, which is now a sprawling tech conglomerate worth $14bn. Mr Son’s SoftBank, after spectacular investments in early internet stocks, muscled into Japan’s telecoms industry. They have both invested heavily in Silicon Valley. They also each own baseball teams named after birds of prey; the SoftBank Hawks and the Rakuten Golden Eagles.
"""
```

For convience, let's check the label dictionary to see which is 'business'.


This can be useful if you have complex labels, or want to quickly reference labels used by the model.


```python
print(flair_model_wrapper.label_dictionary.get_item_for_index(1))

target_label = flair_model_wrapper.label_dictionary.get_item_for_index(1)
```

    business


We also create an empty list to store our attribitions results in order to visualize them using Captum.


```python
visualization_list = []
```

Let's run the Layer Integrated Gradient method on the two paragraphs, and determine what drives the prediction.

As an additional note, the number of steps & the estimation method can have an impact on the attribution.

This [tutorial](https://colab.research.google.com/drive/1pgAbzUF2SzF0BdFtGpJbZPWUOhFxT2NZ#scrollTo=sO0Wr7j6TPOR) even uses 7000 steps!


```python
readable_tokens, word_attributions, delta = interpret_sentence(flair_model_wrapper,
                                                                lig,
                                                                sentence,
                                                                target_label,
                                                                visualization_list,
                                                                n_steps=500,
                                                                estimation_method="gausslegendre",
                                                                internal_batch_size=3)
```

    pred:  1 ( 1.00 ) , delta:  tensor([2.8829], dtype=torch.float64)



```python
visualize_attributions(visualization_list)
```

![Screenshot](word-importance.png)


The tokenizer used by your model will have an impact how the original text is displayed. 

We can also see the importance scores of each token.


```python
word_scores = word_attributions.detach().numpy()
```


```python
ordered_lig = [(readable_tokens[i], word_scores[i]) for i in np.argsort(word_scores)][::-1]
```


```python
ordered_lig
```




    [('investment', 0.6912556656584984),
     (',', 0.3798837938229196),
     ('In', 0.3476725938390601),
     ('.', 0.31968725095155986),
     ('Golden', 0.2094622213371851),
     ('roll', 0.15912006355488764),
     ('Eagle', 0.12119987913236946),
     ('each', 0.11796153579109278),
     ('have', 0.11295847290029525),
     ('interview', 0.06794168798818423),
     (',', 0.0591058601487673),
     ('s', 0.05599717840192191),
     ('kita', 0.04546959026524195),
     ('internet', 0.04273298068470459),
     ('even', 0.0398466819989191),
     ('internet', 0.03650298645706512),
     ('both', 0.035969422144733296),
     ('worth', 0.03288273963161129),
     ('billion', 0.03206918459566223),
     ('muscle', 0.028196057380916115),
     ('banker', 0.026940519313020748),
     ('ed', 0.024515105846343522),
     ('mera', 0.02168594978900262),
     ('after', 0.020827375280079875),
     ('Rak', 0.020516629732796308),
     ('uten', 0.019807849524593118),
     ('School', 0.019248880413689953),
     ('’', 0.01823743842859383),
     ('ed', 0.016867976719556504),
     ('Masa', 0.01644864465371571),
     ('California', 0.016289219855490637),
     ('aires', 0.015730388130484867),
     ('s', 0.015342798705848903),
     ('into', 0.015305456363702709),
     ('acquisition', 0.015071586165743767),
     ('eight', 0.014557915025546738),
     ('Business', 0.014145133579602948),
     ('$', 0.01380955997413895),
     ('junior', 0.013787418601338704),
     ('commerce', 0.013291321768625398),
     ('s', 0.012752441313104157),
     ('says', 0.012666033619872839),
     ('youth', 0.012340050638036332),
     ('University', 0.012268527480874838),
     ('telecom', 0.011779017803216926),
     ('in', 0.011363399070759873),
     ('chy', 0.01122615445445774),
     ('uten', 0.011080935084395543),
     ('Son', 0.010893729680025326),
     (',', 0.010742675382186454),
     ('s', 0.010623088938102009),
     ('word', 0.01058447462960167),
     ('ed', 0.010567861384405866),
     ('14', 0.010348779066862455),
     ('and', 0.010329217533879234),
     ('mad', 0.010323909130579314),
     ('Harvard', 0.010283806665502857),
     ('Son', 0.010199211624985688),
     ('oshi', 0.009931504281767492),
     ('s', 0.009267103871308796),
     ('Mi', 0.009072446716778733),
     ('Harvard', 0.009067310775995574),
     ('de', 0.008728932845230443),
     ('They', 0.008599850640577972),
     ('pre', 0.008444680224718708),
     ('is', 0.008305289101735407),
     ('who', 0.008245410539468744),
     ('which', 0.008062956412325368),
     ('were', 0.008037464360298572),
     ('.', 0.008007655971426572),
     ('1990', 0.007998969075537224),
     ('', 0.007828547421148063),
     ('had', 0.007516185774747121),
     ('Rak', 0.007445138070415522),
     ('they', 0.007419603147035519),
     ('spraw', 0.007291239986265182),
     ('', 0.007274717161741033),
     ('hea', 0.007205550346134348),
     ('ds', 0.007101171930895323),
     ('in', 0.0070461631394168915),
     ('Soft', 0.007030144762845085),
     ('hier', 0.007022593752439835),
     ('years', 0.006967712728516514),
     ('he', 0.0069572940226672545),
     ('since', 0.00692360395539953),
     ('.', 0.0068785487185215885),
     ('men', 0.0068646662898585705),
     ('early', 0.006809332493287538),
     ('', 0.0067551726449196535),
     ('in', 0.006718497449361016),
     ('the', 0.006487402797596461),
     ('he', 0.006448858574234409),
     ('in', 0.006401284623110937),
     ('(', 0.006395227462794027),
     ('s', 0.006306714914398683),
     ('cular', 0.006081438191594103),
     ('common', 0.006045394668931697),
     ('In', 0.0059849630689792375),
     ('bla', 0.0059840947763611756),
     ('', 0.005915387172332866),
     ('s', 0.005847310540612831),
     ('con', 0.005794847475648754),
     ('med', 0.005644492739060562),
     ('y', 0.005557610050420243),
     ('', 0.005506660100009401),
     ('', 0.0054384892671667795),
     ('was', 0.005308141139901876),
     ('cade', 0.005201491413434358),
     ('shared', 0.005054423444923374),
     ('', 0.004993443855398026),
     ('in', 0.0049802669311821875),
     ('in', 0.004967655850714802),
     ('in', 0.004963143163939721),
     ('d', 0.004860684596408859),
     ('s', 0.004855202607347464),
     ('vil', 0.004818176073284607),
     ('lot', 0.0047391235606207135),
     ('leading', 0.004626592868067338),
     ('common', 0.004623559083927996),
     ('America', 0.004620150229643054),
     ('Mi', 0.004533758533143607),
     ('his', 0.004529966124574457),
     ('Valley', 0.00446823196314243),
     ('Ber', 0.004466512049410947),
     ('both', 0.004423582020436731),
     ('after', 0.004383257664637291),
     ('Mi', 0.004212012338226788),
     ('did', 0.004188030225582106),
     ('bir', 0.00409328361921365),
     ('the', 0.004070834304485979),
     ('two', 0.0039345972167746605),
     ('tif', 0.0038621703179599507),
     ('Japan', 0.0038397163217816907),
     ('s', 0.0037920483520681465),
     ('y', 0.003780632909400167),
     ('kita', 0.003774366513221155),
     ('he', 0.003771762917732269),
     ('called', 0.0037426191422556747),
     ('own', 0.003586430227375915),
     ('that', 0.0035650706586045104),
     ('studie', 0.003552041219516141),
     ('Japan', 0.0035312250430994444),
     ('of', 0.003523656263595633),
     ('s', 0.0035233879000030437),
     ('pion', 0.0034925831385909122),
     ('zed', 0.0034752212938575725),
     ('both', 0.003434616890461432),
     ('', 0.0033834641142033305),
     ('now', 0.0033283400622553454),
     (';', 0.0033215394740746113),
     ('corporate', 0.003320926542435578),
     ('Silicon', 0.0032976133672770825),
     ('ing', 0.0032847142893514744),
     ('they', 0.0032842200251681857),
     ('Son', 0.0032789029396651285),
     ('ley', 0.0032784454311866116),
     ('in', 0.003270209519082492),
     ('the', 0.0032672908070280615),
     ('specta', 0.0030765616999852705),
     ('of', 0.00302292063074633),
     ('when', 0.0029515442907606672),
     ('ar', 0.0029068123641705253),
     ('past', 0.002898074811583512),
     ('via', 0.002896899879415438),
     ('baseball', 0.0028702740077341774),
     ('ling', 0.0028645601573820518),
     ('', 0.0028492115648038213),
     ('s', 0.0027951320880350646),
     ('', 0.00275273045307238),
     ('ful', 0.0027307150570583137),
     ('an', 0.002693092683087336),
     ('kita', 0.002663441952852167),
     ('Mr', 0.002659559555204177),
     ('ke', 0.002495981375259566),
     ('ling', 0.002403346769639184),
     ('Mr', 0.0022238940361065697),
     ('not', 0.0021935327342768223),
     ('had', 0.0021863775478481685),
     ('have', 0.002120590369760388),
     ('at', 0.0019759129848322173),
     ('at', 0.0019378525230758592),
     ('to', 0.0018076082157551584),
     ('become', 0.0016106128948978197),
     ('ni', 0.0015913091009027544),
     ('from', 0.001574593567381194),
     ('s', 0.001565629527630727),
     ('', 0.0015514581369722223),
     ('te', 0.0015500935017835197),
     ('d', 0.0015495690825289433),
     ('a', 0.0015090668697854553),
     ('the', 0.0013462287157191457),
     ('Mr', 0.0013062535623861212),
     ('of', 0.0012418545365113134),
     ('', 0.0011875846950978803),
     ('his', 0.0011814334414952717),
     ('They', 0.0011452351919261062),
     ('su', 0.0011410615648642),
     ('', 0.0011157656607743067),
     (':', 0.0010595727622973174),
     ('-', 0.0010391047284604251),
     (',', 0.0010223861125789428),
     ('Japanese', 0.0009203203404191963),
     ('They', 0.000917402812666503),
     ('', 0.0008653044699533966),
     (',', 0.0008413340079068053),
     ('Hir', 0.0008052578557327969),
     ('country', 0.0007524433248957288),
     (';', 0.0007218682707854049),
     ('also', 0.0007201568966878034),
     ('na', 0.0005770646547676047),
     ('en', 0.00047837794464923675),
     ('ni', 0.000460999244491573),
     ('', 0.0003177009326118668),
     ('ni', 0.00028699765696415435),
     ('home', 0.00018501785491875438),
     ('at', 0.00017287050363024804),
     ('pur', 0.00016661798122121426),
     (',', 0.00015171541478122425),
     ('', 8.032220756525267e-05),
     ('the', 4.073573648658189e-05),
     ('the', -4.893855691899412e-05),
     ('', -8.376285364310316e-05),
     ('', -0.0001604729517904657),
     ('.', -0.0003046902997394345),
     ('y', -0.00032303724012665855),
     ('when', -0.0003629536018221893),
     ('', -0.0003692521931513257),
     ('glo', -0.00042524618232307873),
     ('', -0.0006197441859726979),
     ('oshi', -0.0006319269530173338),
     ('a', -0.0006340152764768317),
     ('’', -0.0008023648740014105),
     ('know', -0.0008990374756691048),
     ('a', -0.0009023877257658972),
     ('', -0.0009893454685470553),
     ('baseball', -0.0011675527875120127),
     ('', -0.0014120740757374928),
     ('the', -0.0015106400841716884),
     ('and', -0.0015756393962418117),
     ('', -0.0015906424780605729),
     ('’', -0.0018589011549034506),
     (',', -0.0018817317066202781),
     ('a', -0.001889888983407115),
     ('', -0.0021107956288789766),
     ('so', -0.002248018159511781),
     ('Japan', -0.00243142934160251),
     (',', -0.0027921193406570803),
     (';', -0.0028601921594867408),
     ('e', -0.00286928871365156),
     (',', -0.0030254628651192415),
     ('', -0.003099838770788136),
     ('', -0.003591977033604936),
     ('a', -0.0037188058697374743),
     ('“', -0.004016269300519326),
     ('a', -0.004212456632255352),
     ('eer', -0.00433996733144602),
     (',', -0.004497276003098151),
     ('ught', -0.0048007050075236065),
     ('', -0.005501623659287008),
     ('bn', -0.005688943495254072),
     (')', -0.005852465831749145),
     ('.', -0.005960799875401949),
     ('advice', -0.006275085109650945),
     ('.', -0.00631297342622795),
     ('Mr', -0.006842375130095388),
     ('.', -0.006913767255802578),
     (',', -0.006915083719325077),
     ('”', -0.007477027954397511),
     ('interest', -0.0076082544668049515),
     ('Bank', -0.011196977243745599),
     ('Soft', -0.01156357874882836),
     ('tech', -0.012135051611554438),
     ('teams', -0.0130221274507853),
     ('tech', -0.013710927212653944),
     (',', -0.013870453688225091),
     ('invest', -0.014397186467530253),
     ('industry', -0.015188928793556706),
     ('entrepreneur', -0.0153530663471067),
     ('stock', -0.017791399974575576),
     ('entrepreneur', -0.019747696661586864),
     ('Hawk', -0.0217740834244661),
     ('Bank', -0.032412076457616235),
     ('a', -0.036754974824810535),
     ('both', -0.061404357809590526)]



### Contributions & Suggestions

[Pull requests](https://github.com/robinvanschaik/interpret-flair/pulls) and [issues](https://github.com/robinvanschaik/interpret-flair/issues) are welcome! 


Check out this [discussion](https://github.com/flairNLP/flair/issues/1504) regarding explainable AI & Flair integration.

### Authors

[![DOI](https://zenodo.org/badge/314835538.svg)](https://zenodo.org/badge/latestdoi/314835538)

* [Robin van Schaik](https://github.com/robinvanschaik)

### Acknowledgements

* [Flair](https://github.com/flairNLP/flair) for the text classification training framework.
* [Sentence transformers](https://github.com/UKPLab/sentence-transformers) for great sentence-level language models that can be used in Flair.
* [Huggingface](https://github.com/huggingface/transformers) for a large collection of language models that can be used in Flair.
* [Captum](https://github.com/pytorch/captum) for providing the model interpretation framework.
* This [tutorial](https://captum.ai/tutorials/Bert_SQUAD_Interpret) by the Captum team helped me to get started.
* This [discussion](https://github.com/pytorch/captum/issues/414) regarding Captum & XLM type models was also very insightful.
