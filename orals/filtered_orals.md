### [Open-Vocabulary Customization from CLIP via Data-Free Knowledge Distillation](https://openreview.net/forum?id=1aF2D2CPHi)

**TL;DR:** Could we distill models from CLIP without data to meet customized tasks?  
**Conference:** ICLR 2025 Oral  
**Reason:** The paper addresses customization in vision-language models, relevant to multimodal model adaptation.  
**Abstract:** Vision-language models such as CLIP have demonstrated strong zero-shot performance, but their considerable size and inefficient inference limit customizable deployment for users. While knowledge distillation is a solution, it still requires the original data, which is not always available due to copyrights and privacy concerns. For many users seeking open-vocabulary customization, Data-Free Knowledge Distillation (DFKD) emerges as a promising direction. Upon rethinking DFKD, we find that existing methods fail on CLIP due to their heavy reliance on BatchNorm layers, which are unexpectedly unusable in CLIP. Based on our findings, we adopt image-text matching to achieve DFKD for CLIP, enabling customization based on arbitrary class texts. This involves (i) inversing a surrogate dataset from CLIP based on text prompts; and (ii) distilling a student model from CLIP using the surrogate dataset. Specifically, we introduce style dictionary diversification to enhance the diversity of synthetic images. To prevent uncontrollable semantics introduced by diversification, we propose a class consistency maintaining strategy to ensure the consistency of synthetic images. Based on synthetic images with various styles, we further propose meta knowledge distillation to train the student model with good generalization ability. Moreover, we introduce a simple yet effective method to enable customization based on few example images. Comprehensive experiments showcase the superiority of our approach across twelve customized tasks, achieving a 9.33\% improvement compared to existing DFKD methods.  

---

### [Compositional Entailment Learning for Hyperbolic Vision-Language Models](https://openreview.net/forum?id=3i13Gev2hV)

**TL;DR:** We explore the benefits brought in when using visual-semantic compositional hierarchies for learning hyperbolic representations through unsupervised contrastive training.  
**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a model for vision-language representation leveraging hierarchical structures in multimodal data.  
**Abstract:** Image-text representation learning forms a cornerstone in vision-language models, where pairs of images and textual descriptions are contrastively aligned in a shared embedding space. Since visual and textual concepts are naturally hierarchical, recent work has shown that hyperbolic space can serve as a high-potential manifold to learn vision-language representation with strong downstream performance. In this work, for the first time we show how to fully leverage the innate hierarchical nature of hyperbolic embeddings by looking beyond individual image-text pairs. We propose Compositional Entailment Learning for hyperbolic vision-language models. The idea is that an image is not only described by a sentence but is itself a composition of multiple object boxes, each with their own textual description. Such information can be obtained freely by extracting nouns from sentences and using openly available localized grounding models. We show how to hierarchically organize images, image boxes, and their textual descriptions through contrastive and entailment-based objectives. Empirical evaluation on a hyperbolic vision-language model trained with millions of image-text pairs shows that the proposed compositional learning approach outperforms conventional Euclidean CLIP learning, as well as recent hyperbolic alternatives, with better zero-shot and retrieval generalization and clearly stronger hierarchical performance.  

---

### [Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities](https://openreview.net/forum?id=84pDoCD4lH)

**TL;DR:** We present an evaluation protocol to systematically assess the spatial reasoning capabilities of vision language models, and shed light on the ambiguity and cross-cultural diversity of frame of reference in spatial reasoning.  
**Conference:** ICLR 2025 Oral  
**Reason:** Evaluates vision-language models' spatial reasoning, revealing shortcomings in multimodal understanding and robustness.  
**Abstract:** Spatial expressions in situated communication can be ambiguous, as their meanings vary depending on the frames of reference (FoR) adopted by speakers and listeners. While spatial language understanding and reasoning by vision-language models (VLMs) have gained increasing attention, potential ambiguities in these models are still under-explored. To address this issue, we present the COnsistent Multilingual Frame Of Reference Test (COMFORT), an evaluation protocol to systematically assess the spatial reasoning capabilities of VLMs. We evaluate nine state-of-the-art VLMs using COMFORT. Despite showing some alignment with English conventions in resolving ambiguities, our experiments reveal significant shortcomings of VLMs: notably, the models (1) exhibit poor robustness and consistency, (2) lack the flexibility to accommodate multiple FoRs, and (3) fail to adhere to language-specific or culture-specific conventions in cross-lingual tests, as English tends to dominate other languages. With a growing effort to align vision-language models with human cognitive intuitions, we call for more attention to the ambiguous nature and cross-cultural diversity of spatial reasoning.  

---

### [TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes](https://openreview.net/forum?id=8enWnd6Gp3)

**Conference:** ICLR 2025 Oral  
**Reason:** The paper discusses image-to-3D and text-to-3D generation, indicating a multimodal approach.  
**Abstract:** We introduce TetSphere Splatting, a Lagrangian geometry representation designed for high-quality 3D shape modeling. TetSphere splatting leverages an underused yet powerful geometric primitive -- volumetric tetrahedral meshes. It represents 3D shapes by deforming a collection of tetrahedral spheres, with geometric regularizations and constraints that effectively resolve common mesh issues such as irregular triangles, non-manifoldness, and floating artifacts. Experimental results on multi-view and single-view reconstruction highlight TetSphere splatting's superior mesh quality while maintaining competitive reconstruction accuracy compared to state-of-the-art methods. Additionally, TetSphere splatting demonstrates versatility by seamlessly integrating into generative modeling tasks, such as image-to-3D and text-to-3D generation.  

---

### [Proxy Denoising for Source-Free Domain Adaptation](https://openreview.net/forum?id=FIj9IEPCKr)

**Conference:** ICLR 2025 Oral  
**Reason:** The paper addresses challenges in Vision-Language models, relevant to multimodal adaptation and robustness.  
**Abstract:** Source-Free Domain Adaptation (SFDA) aims to adapt a pre-trained source model to an unlabeled target domain with no access to the source data. Inspired by the success of large Vision-Language (ViL) models in many applications, the latest research has validated ViL's benefit for SFDA by using their predictions as pseudo supervision. However, we observe that ViL's supervision could be noisy and inaccurate at an unknown rate, potentially introducing additional negative effects during adaption. To address this thus-far ignored challenge, we introduce a novel Proxy Denoising (__ProDe__) approach. The key idea is to leverage the ViL model as a proxy to facilitate the adaptation process towards the latent domain-invariant space. Concretely, we design a proxy denoising mechanism to correct ViL's predictions. This is grounded on a proxy confidence theory that models the dynamic effect of proxy's divergence against the domain-invariant space during adaptation. To capitalize the corrected proxy, we further derive a mutual knowledge distilling regularization. Extensive experiments show that ProDe significantly outperforms the current state-of-the-art alternatives under both conventional closed-set setting and the more challenging open-set, partial-set, generalized SFDA, multi-target, multi-source, and test-time settings. Our code and data are available at https://github.com/tntek/source-free-domain-adaptation.  

---

### [MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions](https://openreview.net/forum?id=GGlpykXDCa)

**TL;DR:** A novel multi-table evaluation benchmark that evaluate LLMs' multi-table understanding and reasoning ability.  
**Conference:** ICLR 2025 Oral  
**Reason:** The paper evaluates LLMs on multi-table tasks, addressing multimodal reasoning across different data structures.  
**Abstract:** While large language models (LLMs) have made strides in understanding tabular data, current tabular evaluation benchmarks, such as WikiTableQuestions and WikiSQL, are focus on single-table scenarios, which cannot necessarily reflect the complexity of real-world applications. To bridge this gap, we present a \textbf{M}ulti-table and 
Multi-hop Question Answering (MMQA) dataset to assess LLMs' understanding and reasoning capabilities in handling multi-table tasks. The MMQA dataset demands that models perform multiple inferences by drawing evidence from various tables, which are designed to be connected with each other and require models to identify and utilize relationships such as foreign and primary keys. Then, we introduce a comprehensive evaluation framework that tailors to assess LLMs' capabilities in several aspects including Multi-Table Retrieval, Text-to-SQL Generation, Multi-Table QA, Primary Key Selection, and Foreign Key Selection. 
Finally, we propose a novel multi-table retrieval method that achieves state-of-the-art (SOTA) performance on the MMQA dataset compared to several strong baselines. 
Our experiment results reveal that, compared with human performance, both open-source and commercial LLMs leave significant performance room for improvements in multi-table understanding and reasoning tasks. We believe that the MMQA benchmark will enhance and facilitate LLMs' multi-table capabilities in real-world scenarios.  

---

### [MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models](https://openreview.net/forum?id=HnhNRrLPwm)

**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a benchmark for evaluating interleaved multimodal comprehension in vision-language models.  
**Abstract:** Interleaved multimodal comprehension and generation, enabling models to produce and interpret both images and text in arbitrary sequences, have become a pivotal area in multimodal learning. Despite significant advancements, the evaluation of this capability remains insufficient. Existing benchmarks suffer from limitations in data scale, scope, and evaluation depth, while current evaluation metrics are often costly or biased, lacking in reliability for practical applications. To address these challenges, we introduce MMIE, a large-scale knowledge-intensive benchmark for evaluating interleaved multimodal comprehension and generation in Large Vision-Language Models (LVLMs). MMIE comprises 20K meticulously curated multimodal queries, spanning 3 categories, 12 fields, and 102 subfields, including mathematics, coding, physics, literature, health, and arts. It supports both interleaved inputs and outputs, offering a mix of multiple-choice and open-ended question formats to evaluate diverse competencies. Moreover, we propose a reliable automated evaluation metric, leveraging a scoring model fine-tuned with human-annotated data and systematic evaluation criteria, aimed at reducing bias and improving evaluation accuracy. Extensive experiments demonstrate the effectiveness of our benchmark and metrics in providing a comprehensive evaluation of interleaved LVLMs. Specifically, we evaluate eight LVLMs, revealing that even the best models show significant room for improvement, with most achieving only moderate results. We believe MMIE will drive further advancements in the development of interleaved LVLMs.  

---

### [TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio Motion Embedding and Diffusion Interpolation](https://openreview.net/forum?id=LbEWwJOufy)

**Conference:** ICLR 2025 Oral  
**Reason:** The paper addresses cross-modal alignment between audio and gesture, relevant to multimodal model evaluation.  
**Abstract:** We present TANGO, a framework for generating co-speech body-gesture videos. Given a few-minute, single-speaker reference video and target speech audio, TANGO produces high-fidelity videos with synchronized body gestures. TANGO builds on Gesture Video Reenactment (GVR), which splits and retrieves video clips using a directed graph structure - representing video frames as nodes and valid transitions as edges. We address two key limitations of GVR: audio-motion misalignment and visual artifacts in GAN-generated transition frames. In particular, i) we propose retrieving gestures using latent feature distance to improve cross-modal alignment. To ensure the latent features could effectively model the relationship between speech audio and gesture motion, we implement a hierarchical joint embedding space (AuMoClip); ii) we introduce the diffusion-based model to generate high-quality transition frames. Our diffusion model, Appearance Consistent Interpolation (ACInterp), is built upon AnimateAnyone and includes a reference motion module and homography background flow to preserve appearance consistency between generated and reference videos. By integrating these components into the graph-based retrieval framework, TANGO reliably produces realistic, audio-synchronized videos and outperforms all existing generative and retrieval methods. Our code, pretrained models, and datasets are publicly available at https://github.com/CyberAgentAILab/TANGO.  

---

### [SANA: Efficient High-Resolution Text-to-Image Synthesis with Linear Diffusion Transformers](https://openreview.net/forum?id=N8Oj1XhtYZ)

**TL;DR:** Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed.  
**Conference:** ICLR 2025 Oral  
**Reason:** The paper focuses on text-to-image synthesis, a key multimodal task involving vision and language.  
**Abstract:** We introduce Sana, a text-to-image framework that can efficiently generate images up to 4096$\times$4096 resolution. Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU. Core designs include: (1) Deep compression autoencoder: unlike traditional AEs, which compress images only 8$\times$, we trained an AE that can compress images 32$\times$, effectively reducing the number of latent tokens. (2) Linear DiT: we replace all vanilla attention in DiT with linear attention, which is more efficient at high resolutions without sacrificing quality. (3) Decoder-only text encoder: we replaced T5 with modern decoder-only small LLM as the text encoder and designed complex human instruction with in-context learning to enhance the image-text alignment. (4)  Efficient training and sampling: we propose Flow-DPM-Solver to reduce sampling steps, with efficient caption labeling and selection to accelerate convergence. As a result, Sana-0.6B is very competitive with modern giant diffusion model (e.g. Flux-12B), being 20 times smaller and 100+ times faster in measured throughput. Moreover, Sana-0.6B can be deployed on a 16GB laptop GPU, taking less than 1 second to generate a 1024$\times$1024 resolution image. Sana enables content creation at low cost. Code and model will be publicly released upon publication.  

---

### [PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding](https://openreview.net/forum?id=Q6a9W6kzv5)

**TL;DR:** We propose PhysBench to evaluate VLMs' physical understanding, highlighting their limitations and introducing PhysAgent to enhance VLMs' physical understanding.  
**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a benchmark and framework for enhancing vision-language models in understanding physical phenomena.  
**Abstract:** Understanding the physical world is a fundamental challenge in embodied AI, critical for enabling agents to perform complex tasks and operate safely in real-world environments. While Vision-Language Models (VLMs) have shown great promise in reasoning and task planning for embodied agents, their ability to comprehend physical phenomena remains extremely limited.
To close this gap, we introduce PhysBench, a comprehensive benchmark designed to evaluate VLMs' physical world understanding capability across a diverse set of tasks. 
PhysBench contains 10,002 entries of interleaved video-image-text data, categorized into four major domains: physical object properties, physical object relationships, physical scene understanding, and physics-based dynamics, further divided into 19 subclasses and 8 distinct capability dimensions.
Our extensive experiments, conducted on 75 representative VLMs, reveal that while these models excel in common-sense reasoning, they struggle with understanding the physical world---likely due to the absence of physical knowledge in their training data and the lack of embedded physical priors.
To tackle the shortfall, we introduce PhysAgent, a novel framework that combines the generalization strengths of VLMs with the specialized expertise of vision models, significantly enhancing VLMs' physical understanding across a variety of tasks, including an 18.4\% improvement on GPT-4o.
Furthermore, our results demonstrate that enhancing VLMs' physical world understanding capabilities can help embodied agents such as MOKA.
We believe that PhysBench and PhysAgent offer valuable insights and contribute to bridging the gap between VLMs and physical world understanding. [Project Page is here](https://physbench.github.io/)  

---

### [Generator Matching: Generative modeling with arbitrary Markov processes](https://openreview.net/forum?id=RuP17cJtZo)

**TL;DR:** The core principles of flow matching can be vastly generalized to practically all continuous-time Markov processes using Markov generators, unifying all previous methods and opening the door to new generative models agnostic to data modality.  
**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a framework for constructing multimodal models using generative methods.  
**Abstract:** We introduce Generator Matching, a modality-agnostic framework for generative modeling using arbitrary Markov processes. Generators characterize the infinitesimal evolution of a Markov process, which we leverage for generative modeling in a similar vein to flow matching: we construct conditional generators which generate single data points, then learn to approximate the marginal generator which generates the full data distribution. We show that Generator Matching unifies various generative modeling methods, including diffusion models, flow matching and discrete diffusion models. Furthermore, it expands the design space to new and unexplored Markov processes such as jump processes. Finally, Generator Matching enables the construction of superpositions of Markov generative models and enables the construction of multimodal models in a rigorous manner. We empirically validate our method on image and multimodal generation, e.g. showing that superposition with a jump process improves performance.  

---

### [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://openreview.net/forum?id=SI2hI0frk6)

**TL;DR:** Transfusion is a recipe for training a multi-modal model over discrete and continuous data.  
**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a multi-modal model that combines language and image data for improved performance.  
**Abstract:** We introduce Transfusion, a recipe for training a multi-modal model over discrete and continuous data.
Transfusion combines the language modeling loss function (next token prediction) with diffusion to train a single transformer over mixed-modality sequences.
We pretrain multiple Transfusion models up to 7B parameters from scratch on a mixture of text and image data, establishing scaling laws with respect to a variety of uni- and cross-modal benchmarks.
Our experiments show that Transfusion scales significantly better than quantizing images and training a language model over discrete image tokens.
By introducing modality-specific encoding and decoding layers, we can further improve the performance of Transfusion models, and even compress each image to just 16 patches.
We further demonstrate that scaling our Transfusion recipe to 7B parameters and 2T multi-modal tokens produces a model that can generate images and text on a par with similar scale diffusion models and language models, reaping the benefits of both worlds.  

---

### [Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding](https://openreview.net/forum?id=WOzffPgVjF)

**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a model that improves spatio-temporal video grounding using multimodal video-text pairs.  
**Abstract:** Transformer has attracted increasing interest in spatio-temporal video grounding, or STVG, owing to its end-to-end pipeline and promising result. Existing Transformer-based STVG approaches often leverage a set of object queries, which are initialized simply using zeros and then gradually learn target position information via iterative interactions with multimodal features, for spatial and temporal localization. Despite simplicity, these zero object queries, due to lacking target-specific cues, are hard to learn discriminative target information from interactions with multimodal features in complicated scenarios (e.g., with distractors or occlusion), resulting in degradation. Addressing this, we introduce a novel $\textbf{T}$arget-$\textbf{A}$ware Transformer for $\textbf{STVG}$ ($\textbf{TA-STVG}$), which seeks to adaptively generate object queries via exploring target-specific cues from the given video-text pair, for improving STVG. The key lies in two simple yet effective modules, comprising text-guided temporal sampling (TTS) and attribute-aware spatial activation (ASA), working in a cascade. The former focuses on selecting target-relevant temporal cues from a video utilizing holistic text information, while the latter aims at further exploiting the fine-grained visual attribute information of the object from previous target-aware temporal cues, which is applied for object query initialization. Compared to existing methods leveraging zero-initialized queries, object queries in our TA-STVG, directly generated from a given video-text pair, naturally carry target-specific cues, making them adaptive and better interact with multimodal features for learning more discriminative information to improve STVG. In our experiments on three benchmarks, including HCSTVG-v1/-v2 and VidSTG, TA-STVG achieves state-of-the-art performance and significantly outperforms the baseline, validating its efficacy. Moreover, TTS and ASA are designed for general purpose. When applied to existing methods such as TubeDETR and STCAT, we show substantial performance gains, verifying its generality. Code is released at https://github.com/HengLan/TA-STVG.  

---

### [LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior](https://openreview.net/forum?id=Wr3UuEx72f)

**TL;DR:** A holistic video tokenizer with a learned autoregressive generative prior.  
**Conference:** ICLR 2025 Oral  
**Reason:** LARP enhances video tokenization for autoregressive models, contributing to multimodal model capabilities.  
**Abstract:** We present LARP, a novel video tokenizer designed to overcome limitations in current video tokenization methods for autoregressive (AR) generative models. Unlike traditional patchwise tokenizers that directly encode local visual patches into discrete tokens, LARP introduces a holistic tokenization scheme that gathers information from the visual content using a set of learned holistic queries. This design allows LARP to capture more global and semantic representations, rather than being limited to local patch-level information. Furthermore, it offers flexibility by supporting an arbitrary number of discrete tokens, enabling adaptive and efficient tokenization based on the specific requirements of the task. To align the discrete token space with downstream AR generation tasks, LARP integrates a lightweight AR transformer as a training-time prior model that predicts the next token on its discrete latent space. By incorporating the prior model during training, LARP learns a latent space that is not only optimized for video reconstruction but is also structured in a way that is more conducive to autoregressive generation. Moreover, this process defines a sequential order for the discrete tokens, progressively pushing them toward an optimal configuration during training, ensuring smoother and more accurate AR generation at inference time. Comprehensive experiments demonstrate LARPs strong performance, achieving state-of-the-art FVD on the UCF101 class-conditional video generation benchmark. LARP enhances the compatibility of AR models with videos and opens up the potential to build unified high-fidelity multimodal large language models (MLLMs). Project page:  https://hywang66.github.io/larp/  

---

### [Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping](https://openreview.net/forum?id=X1OfiRYCLn)

**TL;DR:** We develop the first dynamic multimodal evaluation protocol with flexible complexity via Vision-Language Bootstrapping.  
**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a dynamic evaluation protocol for multimodal models, addressing data contamination and performance limitations.  
**Abstract:** Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across multimodal tasks such as visual perception and reasoning, leading to good performance on various multimodal evaluation benchmarks. However, these benchmarks keep a static nature and overlap with the pre-training data, resulting in fixed complexity constraints and data contamination issues. This raises the concern regarding the validity of the evaluation. To address these two challenges, we introduce a dynamic multimodal evaluation protocol called Vision-Language Bootstrapping (VLB). VLB provides a robust and comprehensive assessment for LVLMs with reduced data contamination and flexible complexity. To this end, VLB dynamically generates new visual question-answering samples through a multimodal bootstrapping module that modifies both images and language, while ensuring that newly generated samples remain consistent with the original ones by a judge module. By composing various bootstrapping strategies, VLB offers dynamic variants of existing benchmarks with diverse complexities, enabling the evaluation to co-evolve with the ever-evolving capabilities of LVLMs. Extensive experimental results across multiple benchmarks, including SEEDBench, MMBench, and MME, show that VLB significantly reduces data contamination and exposes performance limitations of LVLMs.  

---

### [Comparing noisy neural population dynamics using optimal transport distances](https://openreview.net/forum?id=cNmu0hZ4CL)

**TL;DR:** We propose using optimal transport distances on stochastic processes to compare noisy neural trajectories.  
**Conference:** ICLR 2025 Oral  
**Reason:** Evaluates dynamics of multimodal models in text-to-image synthesis.  
**Abstract:** Biological and artificial neural systems form high-dimensional neural representations that underpin their computational capabilities. Methods for quantifying geometric similarity in neural representations have become a popular tool for identifying computational principles that are potentially shared across neural systems. These methods generally assume that neural responses are deterministic and static. However, responses of biological systems, and some artificial systems, are noisy and dynamically unfold over time. Furthermore, these characteristics can have substantial influence on a system’s computational capabilities. Here, we demonstrate that existing metrics can fail to capture key differences between neural systems with noisy dynamic responses. We then propose a metric for comparing the geometry of noisy neural trajectories, which can be derived as an optimal transport distance between Gaussian processes. We use the metric to compare models of neural responses in different regions of the motor system and to compare the dynamics of latent diffusion models for text-to-image synthesis.  

---

### [Toward Guidance-Free AR Visual Generation via Condition Contrastive Alignment](https://openreview.net/forum?id=kGvXIlIVLM)

**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a method for improving multimodal visual generation by aligning language and visual content.  
**Abstract:** Classifier-Free Guidance (CFG) is a critical technique for enhancing the sample quality of visual generative models. However, in autoregressive (AR) multi-modal generation, CFG introduces design inconsistencies between language and visual content, contradicting the design philosophy of unifying different modalities for visual AR. Motivated by language model alignment methods, we propose Condition Contrastive Alignment (CCA) to facilitate guidance-free AR visual generation. Unlike guidance methods that alter the sampling process to achieve the ideal sampling distribution, CCA directly fine-tunes pretrained models to fit the same distribution target. Experimental results show that CCA can significantly enhance the guidance-free performance of all tested models with just one epoch of fine-tuning (1% of pretraining epochs) on the pretraining dataset. This largely removes the need for guided sampling in AR visual generation and cuts the sampling cost by half. Moreover, by adjusting training parameters, CCA can achieve trade-offs between sample diversity and fidelity similar to CFG. This experimentally confirms the strong theoretical connection between language-targeted alignment and visual-targeted guidance methods, unifying two previously independent research fields.  

---

### [Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents](https://openreview.net/forum?id=kxnoqaisCT)

**Conference:** ICLR 2025 Oral  
**Reason:** The paper introduces a visual grounding model for GUI agents, addressing multimodal interactions in a novel way.  
**Abstract:** Multimodal large language models (MLLMs) are transforming the capabilities of graphical user interface (GUI) agents, facilitating their transition from controlled simulations to complex, real-world applications across various platforms. However, the effectiveness of these agents hinges on the robustness of their grounding capability. Current GUI agents predominantly utilize text-based representations such as HTML or accessibility trees, which, despite their utility, often introduce noise, incompleteness, and increased computational overhead. In this paper, we advocate a human-like embodiment for GUI agents that perceive the environment entirely visually and directly perform pixel-level operations on the GUI. The key is visual grounding models that can accurately map diverse referring expressions of GUI elements to their coordinates on the GUI across different platforms. We show that a simple recipe, which includes web-based synthetic data and slight adaptation of the LLaVA architecture, is surprisingly effective for training such visual grounding models. We collect the largest dataset for GUI visual grounding so far, containing 10M GUI elements and their referring expressions over 1.3M screenshots, and use it to train UGround, a strong universal visual grounding model for GUI agents. Empirical results on six benchmarks spanning three categories (grounding, offline agent, and online agent) show that 1) UGround substantially outperforms existing visual grounding models for GUI agents, by up to 20\% absolute, and 2) agents with UGround outperform state-of-the-art agents, despite the fact that existing agents use additional text-based input while ours only uses visual perception. These results provide strong support for the feasibility and promises of GUI agents that navigate the digital world as humans do.  

---

### [Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation](https://openreview.net/forum?id=meRCKuUpmc)

**Conference:** ICLR 2025 Oral  
**Reason:** The paper introduces a model that integrates vision and action, addressing multimodal interactions in robotic manipulation.  
**Abstract:** Current efforts to learn scalable policies in robotic manipulation primarily fall into two categories: one focuses on "action," which involves behavior cloning from extensive collections of robotic data, while the other emphasizes "vision," enhancing model generalization by pre-training representations or generative models, also referred to as world models, using large-scale visual datasets. This paper presents an end-to-end paradigm that predicts actions using inverse dynamics models conditioned on the robot's forecasted visual states, named Predictive Inverse Dynamics Models (PIDM). By closing the loop between vision and action, the end-to-end PIDM can be a better scalable action learner. In practice, we use Transformers to process both visual states and actions, naming the model Seer. It is initially pre-trained on large-scale robotic datasets, such as DROID, and can be adapted to real-world scenarios with a little fine-tuning data. Thanks to large-scale, end-to-end training and the continuous synergy between vision and action at each execution step, Seer significantly outperforms state-of-the-art methods across both simulation and real-world experiments. It achieves improvements of 13% on the LIBERO-LONG benchmark, 22% on CALVIN ABC-D, and 43% in real-world tasks. Notably, it demonstrates superior generalization for novel objects, lighting conditions, and environments under high-intensity disturbances. Code and models will be publicly available.  

---

### [ChartMoE: Mixture of Diversely Aligned Expert Connector for Chart Understanding](https://openreview.net/forum?id=o5TsWTUSeF)

**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a model for multimodal chart understanding using a mixture of experts architecture.  
**Abstract:** Automatic chart understanding is crucial for content comprehension and document parsing. Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in chart understanding through domain-specific alignment and fine-tuning. However, current MLLMs still struggle to provide faithful data and reliable analysis only based on charts. To address it, we propose ChartMoE, which employs the Mixture of Expert (MoE) architecture to replace the traditional linear projector to bridge the modality gap. Specifically, we train several linear connectors through distinct alignment tasks, which are utilized as the foundational initialization parameters for different experts. Additionally, we introduce ChartMoE-Align, a dataset with nearly 1 million chart-table-JSON-code quadruples to conduct three alignment tasks (chart-table/JSON/code). Combined with the vanilla connector, we initialize different experts diversely and adopt high-quality knowledge learning to further refine the MoE connector and LLM parameters. Extensive experiments demonstrate the effectiveness of the MoE connector and our initialization strategy, e.g., ChartMoE improves the accuracy of the previous state-of-the-art from 80.48% to 84.64% on the ChartQA benchmark.  

---

### [PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration](https://openreview.net/forum?id=rFpZnn11gj)

**TL;DR:** We present PathGen-1.6M, an open-source large-scale pathology dataset with 1.6M high-quality image-caption pairs, enabling the creation of powerful multimodal models for pathology analysis.  
**Conference:** ICLR 2025 Oral  
**Reason:** The paper introduces a multimodal model for pathology, generating image-text pairs and enhancing analysis capabilities.  
**Abstract:** Vision Language Models (VLMs) like CLIP have attracted substantial attention in pathology, serving as backbones for applications such as zero-shot image classification and Whole Slide Image (WSI) analysis. Additionally, they can function as vision encoders when combined with large language models (LLMs) to support broader capabilities. Current efforts to train pathology VLMs rely on pathology image-text pairs from platforms like PubMed, YouTube, and Twitter, which provide limited, unscalable data with generally suboptimal image quality. In this work, we leverage large-scale WSI datasets like TCGA to extract numerous high-quality image patches. We then train a large multimodal model (LMM) to generate captions for extracted images, creating PathGen-1.6M, a dataset containing 1.6 million high-quality image-caption pairs. Our approach involves multiple agent models collaborating to extract representative WSI patches, generating and refining captions to obtain high-quality image-text pairs. Extensive experiments show that integrating these generated pairs with existing datasets to train a pathology-specific CLIP model, PathGen-CLIP, significantly enhances its ability to analyze pathological images, with substantial improvements across nine pathology-related zero-shot image classification tasks and three whole-slide image tasks. Furthermore, we construct 200K instruction-tuning data based on PathGen-1.6M and integrate PathGen-CLIP with the Vicuna LLM to create more powerful multimodal models through instruction tuning. Overall, we provide a scalable pathway for high-quality data generation in pathology, paving the way for next-generation general pathology models. Our dataset, code, and model are open-access at https://github.com/PathFoundation/PathGen-1.6M.  

---

### [Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective](https://openreview.net/forum?id=tcvMzR2NrP)

**TL;DR:** Through the lens of kinetic optimality, we expand the design space of Discrete Flow Matching, allowing the use of any probability path and simultaneously justifying existing mixture paths.  
**Conference:** ICLR 2025 Oral  
**Reason:** The paper empirically validates a new design space across multiple modalities, including text and image generation.  
**Abstract:** The design space of discrete-space diffusion or flow generative models are significantly less well-understood than their continuous-space counterparts, with many works focusing only on a simple masked construction.
In this work, we aim to take a holistic approach to the construction of discrete generative models based on continuous-time Markov chains, and for the first time, allow the use of arbitrary discrete probability paths, or colloquially, corruption processes. 
Through the lens of optimizing the symmetric kinetic energy, we propose velocity formulas that can be applied to any given probability path, completely decoupling the probability and velocity, and giving the user the freedom to specify any desirable probability path based on expert knowledge specific to the data domain. 
Furthermore, we find that a special construction of mixture probability paths optimizes the symmetric kinetic energy for the discrete case.
We empirically validate the usefulness of this new design space across multiple modalities: text generation, inorganic material generation, and image generation. We find that we can outperform the mask construction even in text with kinetic-optimal mixture paths, while we can make use of domain-specific constructions of the probability path over the visual domain.  

---

### [Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models](https://openreview.net/forum?id=uAFHCZRmXk)

**TL;DR:** We find that an information imbalance between images and texts leads to the modality gap and object bias of contrastive VLMs. We study both phenomena  in depth, eliminate common misconceptions, and improve the understanding of both of them.  
**Conference:** ICLR 2025 Oral  
**Reason:** Analyzes modality gap and object bias in contrastive vision-language models, revealing emergent phenomena in multimodal systems.  
**Abstract:** Contrastive vision-language models (VLMs), like CLIP, have gained popularity for their versatile applicability to various downstream tasks. Despite their successes in some tasks, like zero-shot object recognition, they perform surprisingly poor on other tasks, like attribute recognition. Previous work has attributed these challenges to the modality gap, a separation of image and text in the shared representation space, and to a bias towards objects over other factors, such as attributes. In this analysis paper, we investigate both phenomena thoroughly. We evaluated off-the-shelf VLMs and while the gap's influence on performance is typically overshadowed by other factors, we find indications that closing the gap indeed leads to improvements. Moreover, we find that, contrary to intuition, only few embedding dimensions drive the gap and that the embedding spaces are differently organized. To allow for a clean study of object bias, we introduce a definition and a corresponding measure of it. Equipped with this tool, we find that object bias does not lead to worse performance on other concepts, such as attributes per se. However, why do both phenomena, modality gap and object bias, emerge in the first place? To answer this fundamental question and uncover some of the inner workings of contrastive VLMs, we conducted experiments that allowed us to control the amount of shared information between the modalities. These experiments revealed that the driving factor behind both the modality gap and the object bias, is an information imbalance between images and captions, and unveiled an intriguing connection between the modality gap and entropy of the logits.  

---

### [CyberHost: A One-stage Diffusion Framework for Audio-driven Talking Body Generation](https://openreview.net/forum?id=vaEPihQsAA)

**TL;DR:** We propose a one-stage audio-driven talking body generation framework, CyberHost, designed to produce human videos that match the input audio with high expressiveness and realism.  
**Conference:** ICLR 2025 Oral  
**Reason:** Introduces a model that combines audio and visual modalities for human animation.  
**Abstract:** Diffusion-based video generation technology has advanced significantly, catalyzing a proliferation of research in human animation. While breakthroughs have been made in driving human animation through various modalities for portraits, most of current solutions for human body animation still focus on video-driven methods, leaving audio-driven taking body generation relatively underexplored. In this paper, we introduce CyberHost, a one-stage audio-driven talking body generation framework that addresses common synthesis degradations in half-body animation, including hand integrity, identity consistency, and natural motion.
CyberHost's key designs are twofold. Firstly, the Region Attention Module (RAM) maintains a set of learnable, implicit, identity-agnostic latent features and combines them with identity-specific local visual features to enhance the synthesis of critical local regions. Secondly, the Human-Prior-Guided Conditions introduce more human structural priors into the model, reducing uncertainty in generated motion patterns and thereby improving the stability of the generated videos.
To our knowledge, CyberHost is the first one-stage audio-driven human diffusion model capable of zero-shot video generation for the human body. Extensive experiments demonstrate that CyberHost surpasses previous works in both quantitative and qualitative aspects. CyberHost can also be extended to video-driven and audio-video hybrid-driven scenarios, achieving similarly satisfactory results.  

---

### [Diffusion-Based Planning for Autonomous Driving with Flexible Guidance](https://openreview.net/forum?id=wM2sfVgMDH)

**Conference:** ICLR 2025 Oral  
**Reason:** The paper introduces a model that jointly handles multimodal driving behaviors and emphasizes safety in autonomous driving.  
**Abstract:** Achieving human-like driving behaviors in complex open-world environments is a critical challenge in autonomous driving. Contemporary learning-based planning approaches such as imitation learning methods often struggle to balance competing objectives and lack of safety assurance,due to limited adaptability and inadequacy in learning complex multi-modal behaviors commonly exhibited in human planning, not to mention their strong reliance on the fallback strategy with predefined rules. We propose a novel transformer-based Diffusion Planner for closed-loop planning, which can effectively model multi-modal driving behavior and ensure trajectory quality without any rule-based refinement. Our model supports joint modeling of both prediction and planning tasks under the same architecture, enabling cooperative behaviors between vehicles. Moreover, by learning the gradient of the trajectory score function and employing a flexible classifier guidance mechanism, Diffusion Planner effectively achieves safe and adaptable planning behaviors. Evaluations on the large-scale real-world autonomous planning benchmark nuPlan and our newly collected 200-hour delivery-vehicle driving dataset demonstrate that Diffusion Planner achieves state-of-the-art closed-loop performance with robust transferability in diverse driving styles.  

---

### [Loopy: Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency](https://openreview.net/forum?id=weM4YBicIP)

**TL;DR:** We propose Loopy, an end-to-end audio-conditioned video diffusion model that uses long-term motion information to learn natural motions and improve audio-portrait correlation, eliminating motion constraints and delivering high-quality results.  
**Conference:** ICLR 2025 Oral  
**Reason:** The paper introduces a model that jointly handles audio and video modalities for human motion generation.  
**Abstract:** With the introduction of video diffusion model, audio-conditioned human video generation has recently achieved significant breakthroughs in both the naturalness of motion and the synthesis of portrait details. Due to the limited control of audio signals in driving human motion, existing methods often add auxiliary spatial signals such as movement regions to stabilize movements, which compromise the naturalness and freedom of motion. To address this issue, we propose an end-to-end audio-only conditioned video diffusion model named Loopy. Specifically, we designed two key modules: an inter- and intra-clip temporal module and an audio-to-latents module. These enable the model to better utilize long-term motion dependencies and establish a stronger audio-portrait movement correlation. Consequently, the model can generate more natural and stable portrait videos with subtle facial expressions, without the need for manually setting movement constraints. Extensive experiments show that Loopy outperforms recent audio-driven portrait diffusion models, delivering more lifelike and high-quality results across various scenarios. Video samples are available at https://loopyavataranony.github.io/  

---

### [DSPO: Direct Score Preference Optimization for Diffusion Model Alignment](https://openreview.net/forum?id=xyfb9HHvMe)

**Conference:** ICLR 2025 Oral  
**Reason:** Focuses on aligning multimodal text-to-image diffusion models with human preferences.  
**Abstract:** Diffusion-based Text-to-Image (T2I) models have achieved impressive success in generating high-quality images from textual prompts. While large language models (LLMs) effectively leverage Direct Preference Optimization (DPO) for fine-tuning on human preference data without the need for reward models, diffusion models have not been extensively explored in this area. Current preference learning methods applied to T2I diffusion models immediately adapt existing techniques from LLMs. However, this direct adaptation introduces an estimated loss specific to T2I diffusion models. This estimation can potentially lead to suboptimal performance through our empirical results.  In this work, we  propose Direct Score Preference Optimization (DSPO), a novel algorithm that aligns the pretraining and fine-tuning objectives of diffusion models by leveraging score matching, the same objective used during pretraining. It introduces a new perspective on preference learning for diffusion models. Specifically, DSPO distills the score function of human-preferred image distributions into pretrained diffusion models, fine-tuning the model to generate outputs that align with human preferences. We theoretically show that DSPO shares the same optimization direction as reinforcement learning algorithms in diffusion models under certain conditions. Our experimental results demonstrate that DSPO outperforms preference learning baselines for T2I diffusion models in human preference evaluation tasks and enhances both visual appeal and prompt alignment of generated images.  

---

### [Bayesian-guided Label Mapping for Visual Reprogramming](https://openreview.net/forum?id=135eKqDoRR)

**Conference:** NeurIPS 2024 oral  
**Reason:** The paper evaluates a method for vision-language models, addressing multimodal label mapping.  
**Abstract:** *Visual reprogramming* (VR) leverages the intrinsic capabilities of pretrained vision models by adapting their input or output interfaces to solve downstream tasks whose labels (i.e., downstream labels) might be totally different from the labels associated with the pretrained models (i.e., pretrained labels). 
When adapting the output interface, label mapping methods transform the pretrained labels to downstream labels by establishing a gradient-free one-to-one correspondence between the two sets of labels.
However, in this paper, we reveal that one-to-one mappings may overlook the complex relationship between pretrained and downstream labels. Motivated by this observation, we propose a ***B**ayesian-guided **L**abel **M**apping* (BLM) method. 
BLM constructs an iteratively-updated probabilistic label mapping matrix, with each element quantifying a pairwise relationship between pretrained and downstream labels.
The assignment of values to the constructed matrix is guided by Bayesian conditional probability, considering the joint distribution of the downstream labels and the labels predicted by the pretrained model on downstream samples. Experiments conducted on both pretrained vision models (e.g., ResNeXt) and vision-language models (e.g., CLIP) demonstrate the superior performance of BLM over existing label mapping methods. The success of BLM also offers a probabilistic lens through which to understand and analyze the effectiveness of VR.
Our code is available at https://github.com/tmlr-group/BayesianLM.  

---

### [E2E-MFD: Towards End-to-End Synchronous Multimodal Fusion Detection](https://openreview.net/forum?id=47loYmzxep)

**TL;DR:** A novel end-to-end training algorithm for multimodal fusion detection  
**Conference:** NeurIPS 2024 oral  
**Reason:** Introduces a novel end-to-end algorithm for multimodal fusion detection in autonomous driving.  
**Abstract:** Multimodal image fusion and object detection are crucial for autonomous driving. While current methods have advanced the fusion of texture details and semantic information, their complex training processes hinder broader applications. Addressing this challenge, we introduce E2E-MFD, a novel end-to-end algorithm for multimodal fusion detection. E2E-MFD streamlines the process, achieving high performance with a single training phase. It employs synchronous joint optimization across components to avoid suboptimal solutions associated to individual tasks. Furthermore, it implements a comprehensive optimization strategy in the gradient matrix for shared parameters, ensuring convergence to an optimal fusion detection configuration. Our extensive testing on multiple public datasets reveals E2E-MFD's superior capabilities, showcasing not only visually appealing image fusion but also impressive detection outcomes, such as a 3.9\% and  2.0\% $\text{mAP}_{50}$ increase on horizontal object detection dataset M3FD and oriented object detection dataset DroneVehicle, respectively, compared to state-of-the-art approaches.  

---

### [VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time](https://openreview.net/forum?id=5zSCSE0k41)

**Conference:** NeurIPS 2024 oral  
**Reason:** The paper introduces a model that jointly handles audio and visual modalities for lifelike talking faces.  
**Abstract:** We introduce VASA, a framework for generating lifelike talking faces with appealing visual affective skills (VAS) given a single static image and a speech audio clip. Our premiere model, VASA-1, is capable of not only generating lip movements that are exquisitely synchronized with the audio, but also producing a large spectrum of facial nuances and natural head motions that contribute to the perception of authenticity and liveliness. 
The core innovations include a diffusion-based holistic facial dynamics and head movement generation model that works in a face latent space, and the development of such an expressive and disentangled face latent space using videos.
Through extensive experiments including evaluation on a set of new metrics, we show that our method significantly outperforms previous methods along various dimensions comprehensively. Our method delivers high video quality with realistic facial and head dynamics and also supports the online generation of 512$\times$512 videos at up to 40 FPS with negligible starting latency.
It paves the way for real-time engagements with lifelike avatars that emulate human conversational behaviors.  

---

### [NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video Reconstruction](https://openreview.net/forum?id=8qu52Fl1Dt)

**TL;DR:** This paper proposed NeuroClips, a new state-of-the-art fMRI-to-video reconstruction framework, achieving smooth high-fidelity video reconstruction of up to 6s at 8FPS.  
**Conference:** NeurIPS 2024 oral  
**Reason:** Introduces a model that reconstructs video from fMRI, integrating multiple modalities of brain activity and visual stimuli.  
**Abstract:** Reconstruction of static visual stimuli from non-invasion brain activity fMRI achieves great success, owning to advanced deep learning models such as CLIP and Stable Diffusion. However, the research on fMRI-to-video reconstruction remains limited since decoding the spatiotemporal perception of continuous visual experiences is formidably challenging. We contend that the key to addressing these challenges lies in accurately decoding both high-level semantics and low-level perception flows, as perceived by the brain in response to video stimuli. To the end, we propose NeuroClips, an innovative framework to decode high-fidelity and smooth video from fMRI. NeuroClips utilizes a semantics reconstructor to reconstruct video keyframes, guiding semantic accuracy and consistency, and employs a perception reconstructor to capture low-level perceptual details, ensuring video smoothness. During inference, it adopts a pre-trained T2V diffusion model injected with both keyframes and low-level perception flows for video reconstruction. Evaluated on a publicly available fMRI-video dataset, NeuroClips achieves smooth high-fidelity video reconstruction of up to 6s at 8FPS, gaining significant improvements over state-of-the-art models in various metrics, e.g., a 128% improvement in SSIM and an 81% improvement in spatiotemporal metrics. Our project is available at https://github.com/gongzix/NeuroClips.  

---

### [MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making](https://openreview.net/forum?id=EKdk4vxKO4)

**TL;DR:** MDAgents, a framework that adapts the collaboration of LLMs for complex medical decision-making, improving performance on major medical benchmarks  
**Conference:** NeurIPS 2024 oral  
**Reason:** The paper addresses multi-modal reasoning in medical decision-making using LLMs, relevant to multimodal models.  
**Abstract:** Foundation models are becoming valuable tools in medicine. Yet despite their promise, the best way to leverage Large Language Models (LLMs) in complex medical tasks remains an open question. We introduce a novel multi-agent framework, named **M**edical **D**ecision-making **Agents** (**MDAgents**) that helps to address this gap by automatically assigning a collaboration structure to a team of LLMs. The assigned solo or group collaboration structure is tailored to the medical task at hand, a simple emulation inspired by the way real-world medical decision-making processes are adapted to tasks of different complexities. We evaluate our framework and baseline methods using state-of-the-art LLMs across a suite of real-world medical knowledge and clinical diagnosis benchmarks, including a comparison of
LLMs’ medical complexity classification against human physicians. MDAgents achieved the **best performance in seven out of ten** benchmarks on tasks requiring an understanding of medical knowledge and multi-modal reasoning, showing a significant **improvement of up to 4.2\%** ($p$ < 0.05) compared to previous methods' best performances. Ablation studies reveal that MDAgents effectively determines medical complexity to optimize for efficiency and accuracy across diverse medical tasks. Notably, the combination of moderator review and external medical knowledge in group collaboration resulted in an average accuracy **improvement of 11.8\%**. Our code can be found at https://github.com/mitmedialab/MDAgents.  

---

### [CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://openreview.net/forum?id=TFZlFRl9Ks)

**TL;DR:** CAT3D uses a multi-view diffusion model to create 3D scenes from any number of real or generated images.  
**Conference:** NeurIPS 2024 oral  
**Reason:** Introduces a model that handles multiple input images for 3D scene generation, linking vision modalities.  
**Abstract:** Advances in 3D reconstruction have enabled high-quality 3D capture, but require a user to collect hundreds to thousands of images to create a 3D scene. We present CAT3D, a method for creating anything in 3D by simulating this real-world capture process with a multi-view diffusion model. Given any number of input images and a set of target novel viewpoints, our model generates highly consistent novel views of a scene. These generated views can be used as input to robust 3D reconstruction techniques to produce 3D representations that can be rendered from any viewpoint in real-time. CAT3D can create entire 3D scenes in as little as one minute, and outperforms existing methods for single image and few-view 3D scene creation.  

---

### [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://openreview.net/forum?id=Vi8AepAXGy)

**TL;DR:** Cambrian-1 is a vision-centric study of MLLM design—spanning visual representation choice, connector design, instruction tuning, and benchmarking.  
**Conference:** NeurIPS 2024 oral  
**Reason:** Introduces multimodal LLMs with a focus on vision components and visual grounding.  
**Abstract:** We introduce Cambrian-1, a family of multimodal LLMs (MLLMs) designed with a vision-centric approach. While stronger language models can enhance multimodal capabilities, the design choices for vision components are often insufficiently explored and disconnected from visual representation learning research. This gap hinders accurate sensory grounding in real-world scenarios. Our study uses LLMs and visual instruction tuning as an interface to evaluate various visual representations, offering new insights into different models and architectures—self-supervised, strongly supervised, or combinations thereof—based on experiments with over 15 vision models. We critically examine existing MLLM benchmarks, addressing the difficulties involved in consolidating and interpreting results from various tasks. To further improve visual grounding, we propose spatial vision aggregator (SVA), a dynamic and spatially-aware connector that integrates vision features with LLMs while reducing the number of tokens. Additionally, we discuss the curation of high-quality visual instruction-tuning data from publicly available sources, emphasizing the importance of distribution balancing. Collectively, Cambrian-1 not only achieves state-of-the-art performances but also serves as a comprehensive, open cookbook for instruction-tuned MLLMs. We provide model weights, code, supporting tools, datasets, and detailed instruction-tuning and evaluation recipes. We hope our release will inspire and accelerate advancements in multimodal systems and visual representation learning.  

---

### [Cracking the Code of Juxtaposition: Can AI Models Understand the Humorous Contradictions](https://openreview.net/forum?id=bCMpdaQCNW)

**Conference:** NeurIPS 2024 oral  
**Reason:** The paper evaluates a vision-language model's understanding of multimodal humor, addressing challenges in interpreting complex narratives.  
**Abstract:** Recent advancements in large vision language models have demonstrated remarkable proficiency across a wide range of tasks. 
Yet, these models still struggle with understanding the nuances of human humor through juxtaposition, particularly when it involves nonlinear narratives that underpin many jokes and humor cues.  This paper investigates this challenge by focusing on comics with contradictory narratives, where each comic consists of two panels that create a humorous contradiction. We introduce the YesBut benchmark, which comprises tasks of varying difficulty aimed at assessing AI's capabilities in recognizing and interpreting these comics, ranging from literal content comprehension to deep narrative reasoning. Through extensive experimentation and analysis of recent commercial or open-sourced large vision language models, we assess their capability to comprehend the complex interplay of the narrative humor inherent in these comics. Our results show that even the state-of-the-art models still struggle with this task. Our findings offer insights into the current limitations and potential improvements for AI in understanding human creative expressions.  

---

### [RG-SAN: Rule-Guided Spatial Awareness Network for End-to-End 3D Referring Expression Segmentation](https://openreview.net/forum?id=r5spnrY6H3)

**Conference:** NeurIPS 2024 oral  
**Reason:** The paper addresses multimodal interaction between text and 3D spatial data for segmentation tasks.  
**Abstract:** 3D Referring Expression Segmentation (3D-RES) aims to segment 3D objects by correlating referring expressions with point clouds. However, traditional approaches frequently encounter issues like over-segmentation or mis-segmentation, due to insufficient emphasis on spatial information of instances. In this paper, we introduce a Rule-Guided Spatial Awareness Network (RG-SAN) by utilizing solely the spatial information of the target instance for supervision. This approach enables the network to accurately depict the spatial relationships among all entities described in the text, thus enhancing the reasoning capabilities. The RG-SAN consists of the Text-driven Localization Module (TLM) and the Rule-guided Weak Supervision (RWS) strategy. The TLM initially locates all mentioned instances and iteratively refines their positional information. The RWS strategy, acknowledging that only target objects have supervised positional information, employs dependency tree rules to precisely guide the core instance’s positioning. Extensive testing on the ScanRefer benchmark has shown that RG-SAN not only establishes new performance benchmarks, with an mIoU increase of 5.1 points, but also exhibits significant improvements in robustness when processing descriptions with spatial ambiguity. All codes are available at https://github.com/sosppxo/RG-SAN.  

---

### [MeshFormer : High-Quality Mesh Generation with 3D-Guided Reconstruction Model](https://openreview.net/forum?id=x7pjdDod6Z)

**TL;DR:** We introduce MeshFormer, a sparse-view reconstruction model that can deliver high-quality meshes and be trained efficiently.  
**Conference:** NeurIPS 2024 oral  
**Reason:** The paper integrates 2D and 3D modalities, addressing multimodal input for high-quality mesh generation.  
**Abstract:** Open-world 3D reconstruction models have recently garnered significant attention. However, without sufficient 3D inductive bias, existing methods typically entail expensive training costs and struggle to extract high-quality 3D meshes. In this work, we introduce MeshFormer, a sparse-view reconstruction model that explicitly leverages 3D native structure, input guidance, and training supervision. Specifically, instead of using a triplane representation, we store features in 3D sparse voxels and combine transformers with 3D convolutions to leverage an explicit 3D structure and projective bias. In addition to sparse-view RGB input, we require the network to take input and generate corresponding normal maps. The input normal maps can be predicted by 2D diffusion models, significantly aiding in the guidance and refinement of the geometry's learning. Moreover, by combining Signed Distance Function (SDF) supervision with surface rendering, we directly learn to generate high-quality meshes without the need for complex multi-stage training processes. By incorporating these explicit 3D biases, MeshFormer can be trained efficiently and deliver high-quality textured meshes with fine-grained geometric details. It can also be integrated with 2D diffusion models to enable fast single-image-to-3D and text-to-3D tasks. **Videos are available at https://meshformer3d.github.io/**  

---

### [Multi-granularity Correspondence Learning from Long-term Noisy Videos](https://openreview.net/forum?id=9Cu8MRmhq2)

**Conference:** ICLR 2024 oral  
**Reason:** Addresses multi-granularity correspondence in video-language models, focusing on long-term dependencies and misalignment issues.  
**Abstract:** Existing video-language studies mainly focus on learning short video clips, leaving long-term temporal dependencies rarely explored due to over-high computational cost of modeling long videos. To address this issue, one feasible solution is learning the correspondence between video clips and captions, which however inevitably encounters the multi-granularity noisy correspondence (MNC) problem. To be specific, MNC refers to the clip-caption misalignment (coarse-grained) and frame-word misalignment (fine-grained), hindering temporal learning and video understanding. In this paper, we propose NOise Robust Temporal Optimal traNsport (Norton) that addresses MNC in a unified optimal transport (OT) framework. In brief, Norton employs video-paragraph and clip-caption contrastive losses to capture long-term dependencies based on OT. To address coarse-grained misalignment in video-paragraph contrast, Norton filters out the irrelevant clips and captions through an alignable prompt bucket and realigns asynchronous clip-caption pairs based on transport distance. To address the fine-grained misalignment, Norton incorporates a soft-maximum operator to identify crucial words and key frames. Additionally, Norton exploits the potential faulty negative samples in clip-caption contrast by rectifying the alignment target with OT assignment to ensure precise temporal modeling. Extensive experiments on video retrieval, videoQA, and action segmentation verify the effectiveness of our method. 
Code is available at https://lin-yijie.github.io/projects/Norton.  

---

### [MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts](https://openreview.net/forum?id=KUNzEQMWU7)

**TL;DR:** We introduce MathVista, a novel benchmark for evaluating mathematical reasoning capabilities within visual contexts, and conduct extensive experiments on 11 foundation models.  
**Conference:** ICLR 2024 oral  
**Reason:** Introduces a benchmark for evaluating multimodal models in mathematical reasoning and visual contexts.  
**Abstract:** Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit impressive problem-solving skills in many tasks and domains, but their ability in mathematical reasoning in visual contexts has not been systematically studied. To bridge this gap, we present MathVista, a benchmark designed to combine challenges from diverse mathematical and visual tasks. It consists of 6,141 examples, derived from 28 existing multimodal datasets involving mathematics and 3 newly created datasets (i.e., IQTest, FunctionQA, and PaperQA). Completing these tasks requires fine-grained, deep visual understanding and compositional reasoning, which all state-of-the-art foundation models find challenging. With MathVista, we have conducted a comprehensive, quantitative evaluation of 12 prominent foundation models. The best-performing GPT-4V model achieves an overall accuracy of 49.9%, substantially outperforming Bard, the second-best performer, by 15.1%. Our in-depth analysis reveals that the superiority of GPT-4V is mainly attributed to its enhanced visual perception and mathematical reasoning. However, GPT-4V still falls short of human performance by 10.4%, as it often struggles to understand complex figures and perform rigorous reasoning. This significant gap underscores the critical role that MathVista will play in the development of general-purpose AI agents capable of tackling mathematically intensive and visually rich real-world tasks. We further explore the new ability of self-verification, the application of self-consistency, and the interactive chatbot capabilities of GPT-4V, highlighting its promising potential for future research. The project is available at https://mathvista.github.io/.  

---

### [Unified Generative Modeling of 3D Molecules with Bayesian Flow Networks](https://openreview.net/forum?id=NSVtmmzeRB)

**TL;DR:** A new 3D molecule generative model based on Bayesian Flow Networks  
**Conference:** ICLR 2024 oral  
**Reason:** Introduces a model that handles multiple modalities in 3D molecule generation.  
**Abstract:** Advanced generative model (\textit{e.g.}, diffusion model) derived from simplified continuity assumptions of data distribution, though showing promising progress, has been difficult to apply directly to geometry generation applications due to the \textit{multi-modality} and \textit{noise-sensitive} nature of molecule geometry. 
This work introduces Geometric Bayesian Flow Networks (GeoBFN), which naturally fits molecule geometry by modeling diverse modalities in the differentiable parameter space of distributions. GeoBFN maintains the SE-(3) invariant density modeling property by incorporating equivariant inter-dependency modeling on parameters of distributions and unifying the probabilistic modeling of different modalities. 
Through optimized training and sampling techniques, we demonstrate that GeoBFN achieves state-of-the-art performance on multiple 3D molecule generation benchmarks in terms of generation quality (90.87\% molecule stability in QM9 and 85.6\% atom stability in GEOM-DRUG\footnote{The scores are reported at 1k sampling steps for fair comparison, and our scores could be further improved if sampling sufficiently longer steps.}). GeoBFN can also conduct sampling with any number of steps to reach an optimal trade-off between efficiency and quality (\textit{e.g.}, 20$\times$ speedup without sacrificing performance).  

---

### [Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://openreview.net/forum?id=gU58d5QeGv)

**TL;DR:** We propose an efficient text-to-image model that only requires 1/8th of Stable Diffusion 2.1's compute budget for training and has comparable, if not better image quality with less than half the inference time.  
**Conference:** ICLR 2024 oral  
**Reason:** Focuses on text-to-image synthesis, a multimodal model combining language and vision.  
**Abstract:** We introduce Würstchen, a novel architecture for text-to-image synthesis that combines competitive performance with unprecedented cost-effectiveness for large-scale text-to-image diffusion models.
A key contribution of our work is to develop a latent diffusion technique in which we learn a detailed but extremely compact semantic image representation used to guide the diffusion process. This highly compressed representation of an image provides much more detailed guidance compared to latent representations of language and this significantly reduces the computational requirements to achieve state-of-the-art results. Our approach also improves the quality of text-conditioned image generation based on our user preference study.
The training requirements of our approach consists of 24,602 A100-GPU hours - compared to Stable Diffusion 2.1's 200,000 GPU hours.  
Our approach also requires less training data to achieve these results. Furthermore, our compact latent representations allows us to perform inference over twice as fast, slashing the usual costs and carbon footprint of a state-of-the-art (SOTA) diffusion model significantly, without compromising the end performance. In a broader comparison against SOTA models our approach is substantially more efficient and compares favourably in terms of image quality.
We believe that this work motivates more emphasis on the prioritization of both performance and computational accessibility.  

---

### [Multi-Source Diffusion Models for Simultaneous Music Generation and Separation](https://openreview.net/forum?id=h922Qhkmx1)

**TL;DR:** In this work, we define a diffusion-based generative model which is the first to be capable of both music generation and source separation. We also introduce the partial generation task, where we generate a subset of the sources given the others.  
**Conference:** ICLR 2024 oral  
**Reason:** The paper introduces a model that jointly handles music generation and source separation, involving multimodal audio tasks.  
**Abstract:** In this work, we define a diffusion-based generative model capable of both music generation and source separation by learning the score of the joint probability density of sources sharing a context. Alongside the classic total inference tasks (i.e., generating a mixture, separating the sources), we also introduce and experiment on the partial generation task of source imputation, where we generate a subset of the sources given the others (e.g., play a piano track that goes well with the drums). Additionally, we introduce a novel inference method for the separation task based on Dirac likelihood functions. We train our model on Slakh2100, a standard dataset for musical source separation, provide qualitative results in the generation settings, and showcase competitive quantitative results in the source separation setting. Our method is the first example of a single model that can handle both generation and separation tasks, thus representing a step toward general audio models.  

---

### [Learning Interactive Real-World Simulators](https://openreview.net/forum?id=sFyTZEqmUY)

**TL;DR:** We learn an interactive real-world simulator from broad data rich in different axes that enables long-horizon interactions with humans, vision language models, and reinforcement learning agents.  
**Conference:** ICLR 2024 oral  
**Reason:** Introduces a model that integrates vision and language in a multimodal simulator.  
**Abstract:** Generative models trained on internet data have revolutionized how text, image, and video content can be created. Perhaps the next milestone for generative models is to simulate realistic experience in response to actions taken by humans, robots, and other interactive agents. Applications of a real-world simulator range from controllable content creation in games and movies, to training embodied agents purely in simulation that can be directly deployed in the real world. We explore the possibility of learning a universal simulator (UniSim) of real-world interaction through generative modeling. We first make the important observation that natural datasets available for learning a real-world simulator are often rich along different axes (e.g., abundant objects in image data, densely sampled actions in robotics data, and diverse movements in navigation data). With careful orchestration of diverse datasets, each providing a different aspect of the overall experience, UniSim can emulate how humans and agents interact with the world by simulating the visual outcome of both high-level instructions such as “open the drawer” and low-level controls such as “move by x,y” from otherwise static scenes and objects. There are numerous use cases for such a real-world simulator. As an example, we use UniSim to train both high-level vision-language planners and low-level reinforcement learning policies, each of which exhibit zero-shot real-world transfer after training purely in a learned real-world simulator. We also show that other types of intelligence such as video captioning models can benefit from training with simulated experience in UniSim, opening up even wider applications.  

---

### [Learning to Model the World With Language](https://openreview.net/forum?id=7dP6Yq9Uwv)

**Conference:** ICML 2024 Oral  
**Reason:** The paper introduces a multimodal model that integrates language and visual world understanding.  
**Abstract:** To interact with humans and act in the world, agents need to understand the range of language that people use and relate it to the visual world. While current agents can learn to execute simple language instructions, we aim to build agents that leverage diverse language---language like "this button turns on the TV" or "I put the bowls away"---that conveys general knowledge, describes the state of the world, provides interactive feedback, and more. Our key idea is that *agents should interpret such diverse language as a signal that helps them predict the future*: what they will observe, how the world will behave, and which situations will be rewarded. This perspective unifies language understanding with future prediction as a powerful self-supervised learning objective. We instantiate this in Dynalang, an agent that learns a multimodal world model to predict future text and image representations, and learns to act from imagined model rollouts. While current methods that learn language-conditioned policies degrade in performance with more diverse types of language, we show that Dynalang learns to leverage environment descriptions, game rules, and instructions to excel on tasks ranging from game-playing to navigating photorealistic home scans. Finally, we show that our method enables additional capabilities due to learning a generative model: Dynalang can be pretrained on text-only data, enabling learning from offline datasets, and generate language grounded in an environment.  

---

### [Position: The Platonic Representation Hypothesis](https://openreview.net/forum?id=BH8TYy0r6u)

**Conference:** ICML 2024 Oral  
**Reason:** Discusses convergence across data modalities, relevant to multimodal model behavior.  
**Abstract:** We argue that representations in AI models, particularly deep networks, are converging. First, we survey many examples of convergence in the literature: over time and across multiple domains, the ways by which different neural networks represent data are becoming more aligned. Next, we demonstrate convergence across data modalities: as vision models and language models get larger, they measure distance between datapoints in a more and more alike way. We hypothesize that this convergence is driving toward a shared statistical model of reality, akin to Plato's concept of an ideal reality. We term such a representation the platonic representation and discuss several possible selective pressures toward it. Finally, we discuss the implications of these trends, their limitations, and counterexamples to our analysis.  

---

### [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://openreview.net/forum?id=FPnUhsQJ5B)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a transformer model for text-to-image synthesis, handling two modalities jointly.  
**Abstract:** Diffusion models create data from noise by inverting the forward paths of data towards noise and have emerged as a powerful generative modeling technique for high-dimensional, perceptual data such as images and videos. Rectified flow is a recent generative model formulation that connects data and noise in a straight line. Despite its better theoretical properties and conceptual simplicity, it is not yet decisively established as standard practice. In this work, we improve existing noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales. Through a large-scale study, we demonstrate the superior performance of this approach compared to established diffusion formulations for high-resolution text-to-image synthesis. Additionally, we present a novel transformer-based architecture for text-to-image generation that uses separate weights for the two modalities and enables a bidirectional flow of information between image and text tokens, improving text comprehension, typography, and human preference ratings. We demonstrate that this architecture follows predictable scaling trends and correlates lower validation loss to improved text-to-image synthesis as measured by various metrics and human evaluations. Our largest models outperform state-of-the-art models. Stability AI is considering making experimental data, code, and model weights publicly available.  

---

### [LCA-on-the-Line: Benchmarking Out of Distribution Generalization with Class Taxonomies](https://openreview.net/forum?id=HPXRzM9BYZ)

**Conference:** ICML 2024 Oral  
**Reason:** Analyzes performance of visual-language models, addressing multimodal generalization and OOD challenges.  
**Abstract:** We tackle the challenge of predicting models' Out-of-Distribution (OOD) performance using in-distribution (ID) measurements without requiring OOD data. Existing evaluations with ``Effective robustness'', which use ID accuracy as an indicator of OOD accuracy, encounter limitations when models are trained with diverse supervision and distributions, such as class labels (*Vision Models, VMs, on ImageNet*) and textual descriptions (*Visual-Language Models, VLMs, on LAION*). VLMs often generalize better to OOD data than VMs despite having similar or lower ID performance. To improve the prediction of models' OOD performance from ID measurements, we introduce the *Lowest Common Ancestor (LCA)-on-the-Line* framework. This approach revisits the established concept of LCA distance, which measures the hierarchical distance between labels and predictions within a predefined class hierarchy, such as WordNet. We assess 75 models using ImageNet as the ID dataset and five significantly shifted OOD variants, uncovering a strong linear correlation between ID LCA distance and OOD top-1 accuracy. Our method provides a compelling alternative for understanding why VLMs tend to generalize better. Additionally, we propose a technique to construct a taxonomic hierarchy on any dataset using $K$-means clustering, demonstrating that LCA distance is robust to the constructed taxonomic hierarchy. Moreover, we demonstrate that aligning model predictions with class taxonomies, through soft labels or prompt engineering, can enhance model generalization. Open source code in our [Project Page](https://elvishelvis.github.io/papers/lca/).  

---

### [Image Clustering with External Guidance](https://openreview.net/forum?id=JSYN891WnB)

**Conference:** ICML 2024 Oral  
**Reason:** The paper introduces a method that jointly handles image and text modalities for clustering.  
**Abstract:** The core of clustering lies in incorporating prior knowledge to construct supervision signals. From classic k-means based on data compactness to recent contrastive clustering guided by self-supervision, the evolution of clustering methods intrinsically corresponds to the progression of supervision signals. At present, substantial efforts have been devoted to mining internal supervision signals from data. Nevertheless, the abundant external knowledge such as semantic descriptions, which naturally conduces to clustering, is regrettably overlooked. In this work, we propose leveraging external knowledge as a new supervision signal to guide clustering. To implement and validate our idea, we design an externally guided clustering method (Text-Aided Clustering, TAC), which leverages the textual semantics of WordNet to facilitate image clustering. Specifically, TAC first selects and retrieves WordNet nouns that best distinguish images to enhance the feature discriminability. Then, TAC collaborates text and image modalities by mutually distilling cross-modal neighborhood information. Experiments demonstrate that TAC achieves state-of-the-art performance on five widely used and three more challenging image clustering benchmarks, including the full ImageNet-1K dataset. The code can be accessed at https://github.com/XLearning-SCU/2024-ICML-TAC.  

---

### [VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://openreview.net/forum?id=LRkJwPIDuE)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a model that processes multimodal inputs for video generation.  
**Abstract:** We present VideoPoet, a language model capable of synthesizing high-quality video from a large variety of conditioning signals. VideoPoet employs a decoder-only transformer architecture that processes multimodal inputs -- including images, videos, text, and audio. The training protocol follows that of Large Language Models (LLMs), consisting of two stages: pretraining and task-specific adaptation. During pretraining, VideoPoet incorporates a mixture of multimodal generative objectives within an autoregressive Transformer framework. The pretrained LLM serves as a foundation that can be adapted for a range of video generation tasks. We present empirical results demonstrating the model's state-of-the-art capabilities in zero-shot video generation, specifically highlighting the ability to generate high-fidelity motions. Project page: http://sites.research.google/videopoet/  

---

### [NExT-GPT: Any-to-Any Multimodal LLM](https://openreview.net/forum?id=NZQkumsNlf)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a multimodal model capable of any-to-any content generation across various modalities.  
**Abstract:** While recently Multimodal Large Language Models (MM-LLMs) have made exciting strides, they mostly fall prey to the limitation of only input-side multimodal understanding, without the ability to produce content in multiple modalities. As we humans always perceive the world and communicate with people through various modalities, developing any-to-any MM-LLMs capable of accepting and delivering content in any modality becomes essential to human-level AI. To fill the gap, we present an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT. We connect an LLM with multimodal adaptors and different diffusion decoders, enabling NExT-GPT to perceive inputs and generate outputs in arbitrary combinations of text, image, video, and audio. By leveraging the existing well-trained high-performing encoders and decoders, NExT-GPT is tuned with only a small amount of parameter (1%) of certain projection layers, which not only benefits low-cost training but also facilitates convenient expansion to more potential modalities. Moreover, we introduce a modality-switching instruction tuning (MosIT) and manually curate a high-quality dataset for MosIT, based on which NExT-GPT is empowered with complex cross-modal semantic understanding and content generation. Overall, our research showcases the promising possibility of building a unified AI agent capable of modeling universal modalities, paving the way for more human-like AI research in the community.  

---

### [Video-LaVIT: Unified Video-Language Pre-training with Decoupled Visual-Motional Tokenization](https://openreview.net/forum?id=S9lk6dk4LL)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a model for unified video-language pre-training, handling multiple modalities effectively.  
**Abstract:** In light of recent advances in multimodal Large Language Models (LLMs), there is increasing attention to scaling them from image-text data to more informative real-world videos. Compared to static images, video poses unique challenges for effective large-scale pre-training due to the modeling of its spatiotemporal dynamics. In this paper, we address such limitations in video-language pre-training with an efficient video decomposition that represents each video as keyframes and temporal motions. These are then adapted to an LLM using well-designed tokenizers that discretize visual and temporal information as a few tokens, thus enabling unified generative pre-training of videos, images, and text. At inference, the generated tokens from the LLM are carefully recovered to the original continuous pixel space to create various video content. Our proposed framework is both capable of comprehending and generating image and video content, as demonstrated by its competitive performance across 13 multimodal benchmarks in image and video understanding and generation. Our code and models are available at https://video-lavit.github.io.  

---

### [Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models](https://openreview.net/forum?id=WLPhywf1si)

**Conference:** ICML 2024 Oral  
**Reason:** Provides robustness methods specifically for multimodal models, addressing vulnerabilities in vision-language systems.  
**Abstract:** Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available on GitHub.  

---

### [MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions](https://openreview.net/forum?id=Zc22RDtsvP)

**Conference:** ICML 2024 Oral  
**Reason:** The paper explores multimodal image retrieval using text instructions, addressing joint handling of vision and language.  
**Abstract:** Image retrieval, i.e., finding desired images given a reference image, inherently encompasses rich, multi-faceted search intents that are difficult to capture solely using image-based measures. Recent works leverage text instructions to allow users to more freely express their search intents. However, they primarily focus on image pairs that are visually similar and/or can be characterized by a small set of pre-defined relations. The core thesis of this paper is that text instructions can enable retrieving images with richer relations beyond visual similarity. To show this, we introduce MagicLens, a series of self-supervised image retrieval models that support open-ended instructions. MagicLens is built on a key novel insight: image pairs that naturally occur on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we can bring those implicit relations explicit by synthesizing instructions via foundation models. Trained on 36.7M (query image, instruction, target image) triplets with rich semantic relations mined from the web, MagicLens achieves results comparable with or better than prior best on eight benchmarks of various image retrieval tasks, while maintaining high parameter efficiency with a significantly smaller model size. Additional human analyses on a 1.4M-image unseen corpus further demonstrate the diversity of search intents supported by MagicLens. Code and models are publicly available at the https://open-vision-language.github.io/MagicLens/.  

---

### [Genie: Generative Interactive Environments](https://openreview.net/forum?id=bJbSbJskOS)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a model that generates and interacts with multiple data modalities, including text and images.  
**Abstract:** We introduce Genie, the first *generative interactive environment* trained in an unsupervised manner from unlabelled Internet videos. The model can be prompted to generate an endless variety of action-controllable virtual worlds described through text, synthetic images, photographs, and even sketches. At 11B parameters, Genie can be considered a *foundation world model*. It is comprised of a spatiotemporal video tokenizer, an autoregressive dynamics model, and a simple and scalable latent action model. Genie enables users to act in the generated environments on a frame-by-frame basis *despite training without any ground-truth action labels* or other domain specific requirements typically found in the world model literature. Further the resulting learned latent action space facilitates training agents to imitate behaviors from unseen videos, opening the path for training generalist agents of the future.  

---

### [Contrasting Multiple Representations with the Multi-Marginal Matching Gap](https://openreview.net/forum?id=dV9B9qFeGi)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a loss function for learning representations across multiple modalities, addressing multimodal tasks.  
**Abstract:** Learning meaningful representations of complex objects that can be seen through multiple ($k\geq 3$) views or modalities is a core task in machine learning. Existing methods use losses originally intended for paired views, and extend them to $k$ views, either by instantiating $\tfrac12k(k-1)$ loss-pairs, or by using reduced embeddings, following a *one vs. average-of-rest* strategy. We propose the multi-marginal matching gap (M3G), a loss that borrows tools from multi-marginal optimal transport (MM-OT) theory to simultaneously incorporate all $k$ views. Given a batch of $n$ points, each seen as a $k$-tuple of views subsequently transformed into $k$ embeddings, our loss contrasts the cost of matching these $n$ ground-truth $k$-tuples with the MM-OT polymatching cost, which seeks $n$ optimally arranged $k$-tuples chosen within these $n\times k$ vectors. While the exponential complexity $O(n^k$) of the MM-OT problem may seem daunting, we show in experiments that a suitable generalization of the Sinkhorn algorithm for that problem can scale to, e.g., $k=3\sim 6$ views using mini-batches of size $64~\sim128$. Our experiments demonstrate improved performance over multiview extensions of pairwise losses, for both self-supervised and multimodal tasks.  

---

### [MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark](https://openreview.net/forum?id=dbFEFHAD79)

**Conference:** ICML 2024 Oral  
**Reason:** The paper evaluates multimodal LLMs and highlights challenges like biases and hallucinations in their judgments.  
**Abstract:** Multimodal Large Language Models (MLLMs) have gained significant attention recently, showing remarkable potential in artificial general intelligence. However, assessing the utility of MLLMs presents considerable challenges, primarily due to the absence multimodal benchmarks that align with human preferences. Drawing inspiration from the concept of LLM-as-a-Judge within LLMs, this paper introduces a novel benchmark, termed MLLM-as-a-Judge, to assess the ability of MLLMs in assisting judges across diverse modalities, encompassing three distinct tasks: Scoring Evaluation, Pair Comparison, and Batch Ranking. Our study reveals that, while MLLMs demonstrate remarkable human-like discernment in Pair Comparisons, there is a significant divergence from human preferences in Scoring Evaluation and Batch Ranking tasks. Furthermore, a closer examination reveals persistent challenges in the evaluative capacities of LLMs, including diverse biases, hallucinatory responses, and inconsistencies in judgment, even in advanced models such as GPT-4V. These findings emphasize the pressing need for enhancements and further research efforts to be undertaken before regarding MLLMs as fully reliable evaluators. In light of this, we advocate for additional efforts dedicated to supporting the continuous development within the domain of MLLM functioning as judges. The code and dataset are publicly available at our project homepage: https://mllm-judge.github.io/.  

---

### [Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition](https://openreview.net/forum?id=fO31YAyNbI)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a multimodal model for video reasoning, integrating spatial-temporal understanding with cognitive-level comprehension.  
**Abstract:** Existing research of video understanding still struggles to achieve in-depth comprehension and reasoning in complex videos, primarily due to the under-exploration of two key bottlenecks: fine-grained spatial-temporal perceptive understanding and cognitive-level video scene comprehension. This paper bridges the gap by presenting a novel solution. We first introduce a novel video Multimodal Large Language Model (MLLM), MotionEpic, which achieves fine-grained pixel-level spatial-temporal video grounding by integrating video spatial-temporal scene graph (STSG) representation. Building upon MotionEpic, we then develop a Video-of-Thought (VoT) reasoning framework. VoT inherits the Chain-of-Thought (CoT) core, breaking down a complex task into simpler and manageable sub-problems, and addressing them step-by-step from a low-level pixel perception to high-level cognitive interpretation. Extensive experiments across various complex video QA benchmarks demonstrate that our overall framework strikingly boosts existing state-of-the-art. To our knowledge, this is the first attempt at successfully implementing the CoT technique for achieving human-level video reasoning, where we show great potential in extending it to a wider range of video understanding scenarios. Systems and codes will be open later.  

---

### [SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code](https://openreview.net/forum?id=gAyzjHw2ml)

**Conference:** ICML 2024 Oral  
**Reason:** It involves vision-language integration for 3D scene synthesis, addressing multimodal interactions.  
**Abstract:** This paper introduces SceneCraft, a Large Language Model (LLM) Agent converting text descriptions into Blender-executable Python scripts which render complex scenes with up to a hundred 3D assets. This process requires complex spatial planning and arrangement. We tackle these challenges through a combination of advanced abstraction, strategic planning, and library learning. SceneCraft first models a scene graph as a blueprint, detailing the spatial relationships among assets in the scene. SceneCraft then writes Python scripts based on this graph, translating relationships into numerical constraints for asset layout. Next, SceneCraft leverages the perceptual strengths of vision-language foundation models like GPT-V to analyze rendered images and iteratively refine the scene. On top of this process, SceneCraft features a library learning mechanism that compiles common script functions into a reusable library, facilitating continuous self-improvement without expensive LLM parameter tuning. Our evaluation demonstrates that SceneCraft surpasses existing LLM-based agents in rendering complex scenes, as shown by its adherence to constraints and favorable human assessments. We also showcase the broader application potential of SceneCraft by reconstructing detailed 3D scenes from the Sintel movie and guiding a video generative model with generated scenes as intermediary control signal.  

---

### [Fast Timing-Conditioned Latent Audio Diffusion](https://openreview.net/forum?id=jOlO8t1xdx)

**Conference:** ICML 2024 Oral  
**Reason:** The paper introduces a model that generates audio conditioned on text prompts, addressing multimodal interaction.  
**Abstract:** Generating long-form 44.1kHz stereo audio from text prompts can be computationally demanding. Further, most previous works do not tackle that music and sound effects naturally vary in their duration. Our research focuses on the efficient generation of long-form, variable-length stereo music and sounds at 44.1kHz using text prompts with a generative model. It is based on latent diffusion, with its latent defined by a fully-convolutional variational autoencoder. The generative model is conditioned on text prompts as well as timing embeddings, allowing for fine control over both the content and length of the generated music and sounds. It is capable of rendering stereo signals of up to 95 sec at 44.1kHz in 8 sec on an A100 GPU. Despite its compute efficiency and fast inference, the proposed model is one of the best in two public text-to-music and -audio benchmarks and, differently from state-of-the-art models, can generate music with structure and stereo sounds.  

---

### [FedMBridge: Bridgeable Multimodal Federated Learning](https://openreview.net/forum?id=jrHUbftLd6)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a model for multimodal federated learning addressing diverse modality types and heterogeneity.  
**Abstract:** Multimodal Federated Learning (MFL) addresses the setup of multiple clients with diversified modality types (e.g. image, text, video, and audio) working together to improve their local personal models in a data-privacy manner. Prior MFL works rely on restrictive compositional neural architecture designs to ensure inter-client information sharing via blockwise model aggregation, limiting their applicability in the real-world **Architecture-personalized MFL (AMFL)** scenarios, where clients may have distinguished multimodal interaction strategies and there is no restriction on local architecture design. The key challenge in AMFL is how to automatically and efficiently tackle the two heterogeneity patterns--statistical and architecture heterogeneity--while maximizing the beneficial information sharing among clients. To solve this challenge, we propose **FedMBridge**, which leverages a topology-aware hypernetwork to act as a bridge that can automatically balance and digest the two heterogeneity patterns in a communication-efficient manner. Our experiments on four AMFL simulations demonstrate the efficiency and effectiveness of our proposed approach.  

---

### [Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data](https://openreview.net/forum?id=sBJNokmYuV)

**Conference:** ICML 2024 Oral  
**Reason:** The paper focuses on enhancing vision-language models, directly addressing multimodal model performance.  
**Abstract:** Fine-tuning vision-language models (VLMs) with abundant unlabeled data recently has attracted increasing attention. Existing methods that resort to the pseudolabeling strategy would suffer from heavily incorrect hard pseudolabels when VLMs exhibit low zero-shot performance in downstream tasks. To alleviate this issue, we propose a **C**andidate **P**seudolabel **L**earning method, termed **CPL**, to fine-tune VLMs with suitable candidate pseudolabels of unlabeled data in downstream tasks. The core of our method lies in the generation strategy of candidate pseudolabels, which progressively generates refined candidate pseudolabels by both intra- and inter-instance label selection, based on a confidence score matrix for all unlabeled data. This strategy can result in better performance in true label inclusion and class-balanced instance selection. In this way, we can directly apply existing loss functions to learn with generated candidate psueudolabels. Extensive experiments on nine benchmark datasets with three learning paradigms demonstrate the effectiveness of our method. Our code can be found here.  

---

### [A Touch, Vision, and Language Dataset for Multimodal Alignment](https://openreview.net/forum?id=tFEOOH9eH0)

**Conference:** ICML 2024 Oral  
**Reason:** Introduces a multimodal dataset and model that integrates touch with vision and language.  
**Abstract:** Touch is an important sensing modality for humans, but it has not yet been incorporated into a multimodal generative language model. This is partially due to the difficulty of obtaining natural language labels for tactile data and the complexity of aligning tactile readings with both visual observations and language descriptions. As a step towards bridging that gap, this work introduces a new dataset of 44K in-the-wild visiontouch pairs, with English language labels annotated by humans (10%) and textual pseudo-labels from GPT-4V (90%). We use this dataset to train a vision-language-aligned tactile encoder for open-vocabulary classification and a touch-visionlanguage (TVL) model for text generation using the trained encoder. Results suggest that by incorporating touch, the TVL model improves (+29% classification accuracy) tactile-vision-language alignment over existing models trained on any pair of those modalities. Although only a small fraction of the dataset is human labeled, the TVL model demonstrates improved visual-tactile understanding over GPT-4V (+12%) and open-source vision-language models (+32%) on a new touch-vision understanding benchmark. Code, checkpoints and data are available on https: //tactile-vlm.github.io.  

---

### [GPTSwarm: Language Agents as Optimizable Graphs](https://openreview.net/forum?id=uTC9AFXIhg)

**Conference:** ICML 2024 Oral  
**Reason:** The paper discusses processing multimodal data and optimizing interactions between language agents, relevant to multimodal models.  
**Abstract:** Various human-designed prompt engineering techniques have been proposed to improve problem solvers based on Large Language Models (LLMs), yielding many disparate code bases. We unify these approaches by describing LLM-based agents as computational graphs. The nodes implement functions to process multimodal data or query LLMs, and the edges describe the information flow between operations. Graphs can be recursively combined into larger composite graphs representing hierarchies of inter-agent collaboration (where edges connect operations of different agents). Our novel automatic graph optimizers (1) refine node-level LLM prompts (node optimization) and (2) improve agent orchestration by changing graph connectivity (edge optimization). Experiments demonstrate that our framework can be used to efficiently develop, integrate, and automatically improve various LLM agents. Our code is public.  

---

### [Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models](https://openreview.net/forum?id=0A9f2jZDGW)

**TL;DR:** We present a comprehensive study of task arithmetic and find that linearizing models before fine-tuning improves their performance after editing.  
**Conference:** NeurIPS 2023 oral  
**Reason:** Analyzes task arithmetic in vision-language models, linking model editing to multimodal performance.  
**Abstract:** Task arithmetic has recently emerged as a cost-effective and scalable approach to edit pre-trained models directly in weight space: By adding the fine-tuned weights of different tasks, the model's performance can be improved on these tasks, while negating them leads to task forgetting. Yet, our understanding of the effectiveness of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show that fine-tuning models in their tangent space by linearizing them amplifies weight disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we provide theoretical and empirical analyses of the neural tangent kernel (NTK) of these models and establish a compelling link between task arithmetic and the spatial localization of the NTK eigenfunctions. Overall, our work uncovers novel insights into the fundamental mechanisms of task arithmetic and offers a more reliable and effective approach to edit pre-trained models through the NTK linearization.  

---

### [Image Captioners Are Scalable Vision Learners Too](https://openreview.net/forum?id=A7feCufBhL)

**TL;DR:** We present an extensive comparison of contrastive representation learning and representation learning via image captioning from large image-text data sets.  
**Conference:** NeurIPS 2023 oral  
**Reason:** Analyzes pretraining strategies for multimodal models, specifically image and text.  
**Abstract:** Contrastive pretraining on image-text pairs from the web is one of the most popular large-scale pretraining strategies for vision backbones, especially in the context of large multimodal models. At the same time, image captioning on this type of data is commonly considered an inferior pretraining strategy. In this paper, we perform a fair comparison of these two pretraining strategies, carefully matching training data, compute, and model capacity. Using a standard encoder-decoder transformer, we find that captioning alone is surprisingly effective: on classification tasks, captioning produces vision encoders competitive with contrastively pretrained encoders, while surpassing them on vision & language tasks. We further analyze the effect of the model architecture and scale, as well as the pretraining data on the representation quality, and find that captioning exhibits the same or better scaling behavior along these axes. Overall our results show that plain image captioning is a more powerful pretraining strategy than was previously believed. Code is available at [https://github.com/google-research/big_vision](https://github.com/google-research/big_vision).  

---

### [Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment](https://openreview.net/forum?id=AOKU4nRw1W)

**Conference:** NeurIPS 2023 oral  
**Reason:** Addresses multimodal model issues in text-to-image generation through attention map alignment.  
**Abstract:** Text-conditioned image generation models often generate incorrect associations between entities and their visual attributes. This reflects an impaired mapping between linguistic binding of entities and modifiers in the prompt and visual binding of the corresponding elements in the generated image. As one example, a query like ``a pink sunflower and a yellow flamingo'' may incorrectly produce an image of a yellow sunflower and a pink flamingo. To remedy this issue, we propose SynGen, an approach which first syntactically analyses the prompt to identify entities and their modifiers, and then uses a novel loss function that encourages the cross-attention maps to agree with the linguistic binding reflected by the syntax. Specifically, we encourage large overlap between attention maps of entities and their modifiers, and small overlap with other entities and modifier words. The loss is optimized during inference, without retraining or fine-tuning the model. Human evaluation on three datasets, including one new and challenging set, demonstrate significant improvements of SynGen compared with current state of the art methods. This work highlights how making use of sentence structure during inference can efficiently and substantially improve the faithfulness of text-to-image generation.  

---

### [Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity](https://openreview.net/forum?id=i913TUOvTK)

**TL;DR:** We present Mind-Video for reconstructing continuous video from fMRI data via multimodal contrastive learning and inflated Stable Diffusion model.  
**Conference:** NeurIPS 2023 oral  
**Reason:** The paper introduces a model that reconstructs videos from brain activity, involving multimodal data processing.  
**Abstract:** Reconstructing human vision from brain activities has been an appealing task that helps to understand our cognitive process. Even though recent research has seen great success in reconstructing static images from non-invasive brain recordings, work on recovering continuous visual experiences in the form of videos is limited. In this work, we propose Mind-Video that learns spatiotemporal information from continuous fMRI data of the cerebral cortex progressively through masked brain modeling, multimodal contrastive learning with spatiotemporal attention, and co-training with an augmented Stable Diffusion model that incorporates network temporal inflation. 
We show that high-quality videos of arbitrary frame rates can be reconstructed with Mind-Video using adversarial guidance. The recovered videos were evaluated with various semantic and pixel-level metrics. We achieved an average accuracy of 85% in semantic classification tasks and 0.19 in structural similarity index (SSIM), outperforming the previous state-of-the-art by 45%. We also show that our model is biologically plausible and interpretable, reflecting established physiological processes.  

---

### [Visual Instruction Tuning](https://openreview.net/forum?id=w0H2xGHlkw)

**Conference:** NeurIPS 2023 oral  
**Reason:** Introduces a multimodal model combining vision and language for instruction following.  
**Abstract:** Instruction tuning large language models (LLMs) using machine-generated instruction-following data has been shown to improve zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. We present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and an LLM for general-purpose visual and language understanding. To facilitate future research on visual instruction following, we construct two evaluation benchmarks with diverse and challenging application-oriented tasks. Our experiments show that LLaVA demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model, and code publicly available.  

---

### Pre-training for Speech Translation: CTC Meets Optimal Transport

**Conference:** ICML 2023 OralPoster  
**Reason:** The paper addresses the speech and text modalities, focusing on improving speech-to-text translation.  
**Abstract:** The gap between speech and text modalities is a major challenge in speech-to-text translation (ST). Different methods have been proposed to reduce this gap, but most of them require architectural changes in ST training. In this work, we propose to mitigate this issue at the pre-training stage, requiring no change in the ST model. First, we show that the connectionist temporal classification (CTC) loss can reduce the modality gap by design. We provide a quantitative comparison with the more common cross-entropy loss, showing that pre-training with CTC consistently achieves better final ST accuracy. Nevertheless, CTC is only a partial solution and thus, in our second contribution, we propose a novel pre-training method combining CTC and optimal transport to further reduce this gap. Our method pre-trains a Siamese-like model composed of two encoders, one for acoustic inputs and the other for textual inputs, such that they produce representations that are close to each other in the Wasserstein space. Extensive experiments on the standard CoVoST-2 and MuST-C datasets show that our pre-training method applied to the vanilla encoder-decoder Transformer achieves state-of-the-art performance under the no-external-data setting, and performs on par with recent strong multi-task learning systems trained with external data. Finally, our method can also be applied on top of these multi-task systems, leading to further improvements for these models.  

---

### Cross-Modal Fine-Tuning: Align then Refine

**Conference:** ICML 2023 OralPoster  
**Reason:** Introduces a cross-modal fine-tuning framework for diverse modalities, addressing multimodal model alignment.  
**Abstract:** Fine-tuning large-scale pretrained models has led to tremendous progress in well-studied modalities such as vision and NLP. However, similar gains have not been observed in many other modalities due to a lack of relevant pretrained models. In this work, we propose ORCA, a general cross-modal fine-tuning framework that extends the applicability of a single large-scale pretrained model to diverse modalities. ORCA adapts to a target task via an align-then-refine workflow: given the target input, ORCA first learns an embedding network that aligns the embedded feature distribution with the pretraining modality. The pretrained model is then fine-tuned on the embedded data to exploit the knowledge shared across modalities. Through extensive experiments, we show that ORCA obtains state-of-the-art results on 3 benchmarks containing over 60 datasets from 12 modalities, outperforming a wide range of hand-designed, AutoML, general-purpose, and task-specific cross-modal methods. We highlight the importance of data alignment via a series of ablation studies and exemplify ORCA's utility in data-limited regimes.  

---

### Calibrating Multimodal Learning

**Conference:** ICML 2023 OralPoster  
**Reason:** Addresses reliability and robustness in multimodal models, focusing on confidence calibration.  
**Abstract:** Multimodal machine learning has achieved remarkable progress in a wide range of scenarios. However, the reliability of multimodal learning remains largely unexplored. In this paper, through extensive empirical studies, we identify current multimodal classification methods suffer from unreliable predictive confidence that tend to rely on partial modalities when estimating confidence. Specifically, we find that the confidence estimated by current models could even increase when some modalities are corrupted. To address the issue, we introduce an intuitive principle for multimodal learning, i.e., the confidence should not increase when one modality is removed. Accordingly, we propose a novel regularization technique, i.e., Calibrating Multimodal Learning (CML) regularization, to calibrate the predictive confidence of previous methods. This technique could be flexibly equipped by existing models and improve the performance in terms of confidence calibration, classification accuracy, and model robustness.  

---

### Reparameterized Policy Learning for Multimodal Trajectory Optimization

**Conference:** ICML 2023 OralPoster  
**Reason:** Introduces a multimodal policy for reinforcement learning, addressing challenges in high-dimensional action spaces.  
**Abstract:** We investigate the challenge of parametrizing policies for reinforcement learning (RL) in high-dimensional continuous action spaces. Our objective is to develop a multimodal policy that overcomes limitations inherent in the commonly-used Gaussian parameterization. To achieve this, we propose a principled framework that models the continuous RL policy as a generative model of optimal trajectories. By conditioning the policy on a latent variable, we derive a novel variational bound as the optimization objective, which promotes exploration of the environment. We then present a practical model-based RL method, called Reparameterized Policy Gradient (RPG), which leverages the multimodal policy parameterization and learned world model to achieve strong exploration capabilities and high data efficiency. Empirical results demonstrate that our method can help agents evade local optima in tasks with dense rewards and solve challenging sparse-reward environments by incorporating an object-centric intrinsic reward. Our method consistently outperforms previous approaches across a range of tasks. Code and supplementary materials are available on the project page https://haosulab.github.io/RPG/  

---

### Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

**Conference:** ICML 2023 OralPoster  
**Reason:** The paper evaluates pruning methods on multimodal models like CLIP, addressing efficiency in large-scale multimodal systems.  
**Abstract:** Large pre-trained transformers have been receiving explosive attention in the past few years, due to their acculturation for numerous downstream applications via fine-tuning, but their exponentially increasing parameter counts are becoming a primary hurdle to even just fine-tune them without industry-standard hardware. Recently, Lottery Ticket Hypothesis (LTH) and its variants, have been exploited to prune these large pre-trained models generating subnetworks which can achieve similar performance as their dense counterparts, but LTH pragmatism is enormously inhibited by repetitive full training and pruning routine of iterative magnitude pruning (IMP) which worsens with increasing model size. Motivated by the recent observations of model soups, which suggest that fine-tuned weights of multiple models can be merged to a better minima, we propose **Instant Soup Pruning (ISP)** to generate lottery ticket quality subnetworks, using a fraction of the original IMP cost by replacing the expensive intermediate pruning stages of IMP with computationally efficient weak mask generation and aggregation routine. More specifically, during the mask generation stage, ISP takes a small handful of iterations using varying training protocols and data subsets to generate many weak and noisy subnetworks, and superpose them to average out the noise creating a high-quality denoised subnetwork. Our extensive experiments and ablation on two popular large-scale pre-trained models: $\texttt{CLIP} (unexplored in pruning till date)$ and $\texttt{BERT}$ across multiple benchmark vision $\texttt{\{MNIST, SVHN, Cars, GTSRB, CIFAR-10, CIFAR-100\}}$ and language datasets $\texttt{\{MNLI, QNLI, QQP, SST, ...\}}$ validate the effectiveness of ISP compared to several state-of-the-art pruning methods. Additionally, we show that ISP can be easily modified with minimal overhead to produce benefits comparable to model soups, without the prerequisite to generate multiple candidates fine-tuned models. Codes are available at: https://github.com/VITA-Group/instant_soup.  

---

### Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language

**Conference:** ICML 2023 OralPoster  
**Reason:** The paper addresses self-supervised learning across multiple modalities: vision, speech, and language.  
**Abstract:** Current self-supervised learning algorithms are often modality-specific and require large amounts of computational resources. To address these issues, we increase the training efficiency of data2vec, a learning objective that generalizes across several modalities. We do not encode masked tokens, use a fast convolutional decoder and amortize the effort to build teacher representations. data2vec 2.0 benefits from the rich contextualized target representations introduced in data2vec which enable a fast self-supervised learner. Experiments on ImageNet-1K image classification show that data2vec 2.0 matches the accuracy of Masked Autoencoders in 16.4x lower pre-training time, on Librispeech speech recognition it performs as well as wav2vec 2.0 in 10.6x less time, and on GLUE natural language understanding it matches a retrained RoBERTa model in half the time. Trading some speed for accuracy results in ImageNet-1K top-1 accuracy of 86.8% with a ViT-L model trained for 150 epochs.  

---

### TRAK: Attributing Model Behavior at Scale

**Conference:** ICML 2023 OralPoster  
**Reason:** Evaluates a method for data attribution in vision-language models, relevant to multimodal systems.  
**Abstract:** The goal of *data attribution* is to trace model predictions back to training data. Despite a long line of work towards this goal, existing approaches to data attribution tend to force users to choose between computational tractability and efficacy. That is, computationally tractable methods can struggle with accurately attributing model predictions in non-convex settings (e.g., in the context of deep neural networks), while methods that are effective in such regimes require training thousands of models, which makes them impractical for large models or datasets. In this work, we introduce TRAK (Tracing with the Randomly-projected After Kernel), a data attribution method that is both effective *and* computationally tractable for large-scale, differentiable models. In particular, by leveraging only a handful of trained models, TRAK can match the performance of attribution methods that require training thousands of models. We demonstrate the utility of TRAK across various modalities and scales: image classifiers trained on ImageNet, vision-language models (CLIP), and language models (BERT and mT5). We provide code for using TRAK (and reproducing our work) at https://github.com/MadryLab/trak .  

---

### StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis

**Conference:** ICML 2023 OralPoster  
**Reason:** Focuses on text-to-image synthesis, a key multimodal task involving vision and language.  
**Abstract:** Text-to-image synthesis has recently seen significant progress thanks to large pretrained language models, large-scale training data, and the introduction of scalable model families such as diffusion and autoregressive models. However, the best-performing models require iterative evaluation to generate a single sample. In contrast, generative adversarial networks (GANs) only need a single forward pass. They are thus much faster, but they currently remain far behind the state-of-the-art in large-scale text-to-image synthesis. This paper aims to identify the necessary steps to regain competitiveness. Our proposed model, StyleGAN-T, addresses the specific requirements of large-scale text-to-image synthesis, such as large capacity, stable training on diverse datasets, strong text alignment, and controllable variation vs. text alignment tradeoff. StyleGAN-T significantly improves over previous GANs and outperforms distilled diffusion models - the previous state-of-the-art in fast text-to-image synthesis - in terms of sample quality and speed.  

---

### ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts

**Conference:** ICML 2023 OralPoster  
**Reason:** Introduces a multimodal framework combining protein sequences and biomedical texts for enhanced representation learning.  
**Abstract:** Current protein language models (PLMs) learn protein representations mainly based on their sequences, thereby well capturing co-evolutionary information, but they are unable to explicitly acquire protein functions, which is the end goal of protein representation learning. Fortunately, for many proteins, their textual property descriptions are available, where their various functions are also described. Motivated by this fact, we first build the ProtDescribe dataset to augment protein sequences with text descriptions of their functions and other important properties. Based on this dataset, we propose the ProtST framework to enhance Protein Sequence pre-training and understanding by biomedical Texts. During pre-training, we design three types of tasks, i.e., unimodal mask prediction, multimodal representation alignment and multimodal mask prediction, to enhance a PLM with protein property information with different granularities and, at the same time, preserve the PLM's original representation power. On downstream tasks, ProtST enables both supervised learning and zero-shot prediction. We verify the superiority of ProtST-induced PLMs over previous ones on diverse representation learning benchmarks. Under the zero-shot setting, we show the effectiveness of ProtST on zero-shot protein classification, and ProtST also enables functional protein retrieval from a large-scale database without any function annotation.  

---

### Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding

**Conference:** ICML 2023 OralPoster  
**Reason:** Introduces a model that integrates vision and language for visual language understanding.  
**Abstract:** Visually-situated language is ubiquitous---sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and forms. Perhaps due to this diversity, previous work has typically relied on domain-specific recipes with limited sharing of the underlying data, model architectures, and objectives. We present Pix2Struct, a pretrained image-to-text model for purely visual language understanding, which can be finetuned on tasks containing visually-situated language. Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML. The web, with its richness of visual elements cleanly reflected in the HTML structure, provides a large source of pretraining data well suited to the diversity of downstream tasks. Intuitively, this objective subsumes common pretraining signals such as OCR, language modeling, and image captioning. In addition to the novel pretraining strategy, we introduce a variable-resolution input representation and a more flexible integration of language and vision inputs, where language prompts such as questions are rendered directly on top of the input image. For the first time, we show that a single pretrained model can achieve state-of-the-art results in six out of nine tasks across four domains: documents, illustrations, user interfaces, and natural images.  

---

### Mu$^2$SLAM: Multitask, Multilingual Speech and Language Models

**Conference:** ICML 2023 OralPoster  
**Reason:** Introduces a model that jointly handles speech and text, addressing multimodal representation alignment.  
**Abstract:** We present Mu$^2$SLAM, a multilingual sequence-to-sequence model pre-trained jointly on unlabeled speech, unlabeled text and supervised data spanning Automatic Speech Recognition (ASR), Automatic Speech Translation (AST) and Machine Translation (MT), in over 100 languages. By leveraging a quantized representation of speech as a target, Mu$^2$SLAM trains the speech-text models with a sequence-to-sequence masked denoising objective similar to T5 on the decoder and a masked language modeling objective (MLM) on the encoder, for both unlabeled speech and text, while utilizing the supervised tasks to improve cross-lingual and cross-modal representation alignment within the model. On CoVoST AST, Mu$^2$SLAM establishes a new state-of-the-art for models trained on public datasets, improving on xx-en translation over the previous best by 1.9 BLEU points and on en-xx translation by 1.1 BLEU points. On Voxpopuli ASR, our model matches the performance of an mSLAM model fine-tuned with an RNN-T decoder, despite using a relatively weaker Transformer decoder. On text understanding tasks, our model improves by more than 6% over mSLAM on XNLI, getting closer to the performance of mT5 models of comparable capacity on XNLI and TydiQA, paving the way towards a single model for all speech and text understanding tasks.  

---

### Information-Theoretic State Space Model for Multi-View Reinforcement Learning

**Conference:** ICML 2023 OralPoster  
**Reason:** The paper addresses multi-view observations, relevant to multimodal data handling in reinforcement learning.  
**Abstract:** Multi-View Reinforcement Learning (MVRL) seeks to find an optimal control for an agent given multi-view observations from various sources. Despite recent advances in multi-view learning that aim to extract the latent representation from multi-view data, it is not straightforward to apply them to control tasks, especially when the observations are temporally dependent on one another. The problem can be even more challenging if the observations are intermittently missing for a subset of views. In this paper, we introduce Fuse2Control (F2C), an information-theoretic approach to capturing the underlying state space model from the sequences of multi-view observations. We conduct an extensive set of experiments in various control tasks showing that our method is highly effective in aggregating task-relevant information across many views, that scales linearly with the number of views while retaining robustness to arbitrary missing view scenarios.  

---

