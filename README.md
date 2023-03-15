## Python modules required
```
PyTorch
NumPy
libROSA
progressbar2
```

## 1. Prepare data for training

#### 1.1 Generate the RIRs for microphone pairs
According to instruction in the folder `./octave/rir`, we use the room simulator to generate the Room impulse responses(RIRs) for microphone pairs which are used in training.

#### 1.2 Generate list of speech files

Assuming the root directory with librispeech corresponds to `<root_speech>` and we want to store the results in `<speech_meta>`.

```
python3 plan_speech.py --root <root_speech> --json json/speech.json > <speech_meta>
```

#### 1.3 Generate list of farfield files

Assuming the root directory with generated RIRs correspond to `<root_farfield>` and we want to store the results in `<farfield_meta>`.

```
python3 plan_farfield.py --root <root_farfield> --json json/farfield.json > <farfield_meta>
```

#### 1.4 Generate list of audio files

Suppose we want to generate 10,000 samples and store the list in `<audio_meta>`.

```
python3 plan_audio.py --speech <speech_meta> --farfield <farfield_meta> --count 10000 > <audio_meta>
```

## 2. Train model

#### 2.1 Initialize model

Create a model with random parameters to start training from, and save the model in the file `<model_init>`.

```
python3 train_init.py --json json/features.json --model_dst <model_init>
```

#### 2.2 Train over epochs

Train over one epoch a previous model `<model_prev>` and save the updated version to `<model_next>`.

```
python3 train_epochs.py --audio <audio_meta> --json json/features.json --model_src <model_prev> --model_dst <model_next>
```

## 3. Evaluate with model

#### 3.1 Evaluate a single sample and show result

Evaluate the sample at index `<sample_index>` from the dataset `<audio_meta>` using the trained model saved in the file `<model_trained>`.

```
python3 eval_sample.py --audio <audio_meta> --json json/features.json --model_src <model_trained> --index <sample_index>
```

#### 3.2 Evaluate loss for the whole dataset

Evaluate the mean loss for a given model `<model_trained>` and the dataset `<audio_meta>`.

```
python3 eval_epochs.py --audio <audio_meta> --json json/features.json --model_src <model_trained>
```

## 4. Prepare data for testing 

#### 4.1 Generate the RIRs for other microphone array geometries
According to instruction in the folder `./octave/rir`, we use the room simulator to generate the Room impulse responses(RIRs) for other microphone array geometries which are used in training.

#### 4.2 Generate list of speech files, farfield files and audio files
Do the rest steps which are as same as the steps in the section 1 to generate the list of speech files, farfield files and audio files  


## 5. Perform Maximum SNR beamforming

#### 5.1 Apply beamformer on a single sample and store the result
Apply the Maximum SNR beamforming for the sample at index `<sample_index>` from the dataset `<audio_meta>` using the trained model saved in the file `<model_trained>`, and save the result in a wave file `<wave_output>`, with three-channels, where channel 1 is the reference signal, channel 2 the mixed signals, and channel 3 the processed signal.

```
python3 test_sample.py --audio <audio_meta> --json json/features.json --model_src <model_trained> --wave_dst <wave_output> --index <sample_index>
```

#### 5.2 Apply beamformer on multiple samples and store the result
Apply the Maximum SNR beamforming for the multiple samples with number `<number_elements>` from the dataset `<audio_meta>` using the trained model saved in the file `<model_trained>`, and then save the result in a folder `<wave_output_folder>`, where the wave file is with three-channels in form as previously described.

```
python3 test_elements.py --audio <audio_meta> --json json/features.json --model_src <model_trained> --wave_dst <wave_output> --num_eles <number_elements>
```

## 6. Evaluate the performance of beamformer

#### 6.1 Evaluate the improvment of SDR 
Use the BSS Eval toolbox which located in folder `./Octave/bss` to evaluate the mean value of improvment SDR for the number `<number_elements>` of wave files from result of beamformer, which stored in the folder `<wave_output_folder>`

```
octave eval_SDR.m <wave_output_folder>  <number_elements>
```

#### 6.2 Evaluate the improvment of SIR 
Use the BSS Eval toolbox which located in folder `./Octave/bss` to evaluate the mean value of improvment SDR for the number `<number_elements>` of wave files from result of beamformer, which stored in the folder `<wave_output_folder>`

```
octave eval_SIR.m <wave_output_folder>  <number_elements>
```

#### 6.3 Evaluate the improvement of PESQ 
Evaluate the mean value of improvment in PESQ over the number `<number_elements>` of wave files from result of beamforer, which stored in the folder `<wave_output_folder>`

```
python3 cal_pesq.py --wave_dst <wave_output_folder> --num_eles <number_elements>
```