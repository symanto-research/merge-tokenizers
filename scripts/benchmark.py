from collections import defaultdict
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from merge_tokenizers import (
    Aligner,
    DTWAligner,
    FastDTWAligner,
    GreedyCoverageAligner,
    GreedyDistanceAligner,
    PythonDTWAligner,
    PythonGreedyCoverageAligner,
    TamuheyAligner,
    WordIdsAligner,
)
from merge_tokenizers.types import TokenizedSet

texts = [
    """Bradley Charles Cooper (born January 5, 1975) is an American actor and filmmaker. He is the recipient of various accolades, including a British Academy Film Award and two Grammy Awards, in addition to nominations for twelve Academy Awards, six Golden Globe Awards, and a Tony Award. Cooper appeared on the Forbes Celebrity 100 three times and on Time's list of the 100 most influential people in the world in 2015. His films have grossed $13 billion worldwide and he has placed four times in annual rankings of the world's highest-paid actors.
Cooper enrolled in the MFA program at the Actors Studio in 2000 after beginning his career in 1999 with a guest role in the television series Sex and the City. He made his film debut in the comedy Wet Hot American Summer (2001), and gained some recognition as Will Tippin in the television series Alias (2001–2006). After his role in the show was demoted, he began to have career doubts but gained some recognition with a supporting part in the comedy film Wedding Crashers (2005). He had his breakthrough in The Hangover (2009), a critically and commercially successful comedy that spawned two sequels in 2011 and 2013, and his career progressed with starring roles in Limitless (2011) and The Place Beyond the Pines (2012).
Cooper found greater success with the romantic comedy Silver Linings Playbook (2012), the black comedy American Hustle (2013), and the war biopic American Sniper (2014), which he also produced. In 2014, he portrayed Joseph Merrick in a Broadway revival of The Elephant Man and began voicing Rocket in the Marvel Cinematic Universe. In 2018, Cooper produced, wrote, directed, and starred in the musical romance A Star Is Born. He won a BAFTA Award and two Grammys for his contributions to its U.S. Billboard 200 number one soundtrack and its chart-topping lead single "Shallow". He has since produced the thrillers Joker (2019) and Nightmare Alley (2021), and directed the biographical drama Maestro (2023), in which he also starred as Leonard Bernstein.
Cooper was named People magazine's Sexiest Man Alive in 2011. He supports several charities that help fight cancer. Cooper was briefly married to actress Jennifer Esposito, and has a daughter from his relationship with model Irina Shayk.""",
    """config (DebertaConfig) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the from_pretrained() method to load the model weights.
The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top. The DeBERTa model was proposed in DeBERTa: Decoding-enhanced BERT with Disentangled Attention by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It’s build on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.
This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.""",
    """Barack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African-American president in U.S. history. Obama previously served as a U.S. senator representing Illinois from 2005 to 2008, as an Illinois state senator from 1997 to 2004, and as a civil rights lawyer and university lecturer.
Obama was born in Honolulu, Hawaii. He graduated from Columbia University in 1983 with a B.A. in political science and later worked as a community organizer in Chicago. In 1988, Obama enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. He became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. He also went into elective politics. Obama represented the 13th district in the Illinois Senate from 1997 until 2004, when he successfully ran for the U.S. Senate. In 2008, after a close primary campaign against Hillary Clinton, he was nominated by the Democratic Party for president and chose Delaware Senator Joe Biden as his running mate. Obama was elected president, defeating Republican Party nominee John McCain in the presidential election and was inaugurated on January 20, 2009. Nine months later he was named the 2009 Nobel Peace Prize laureate, a decision that drew a mixture of praise and criticism.""",
    """La presidencia de Barack Obama dio comienzo al mediodía del martes 20 de enero de 2009 (EST), tras haber jurado el cargo como el cuadragésimo cuarto presidente de los Estados Unidos durante una ceremonia celebrada en el Capitolio de los Estados Unidos. La misma culminó al mediodía del viernes 20 de enero de 2017, cuando el presidente electo Donald J. Trump asumió la presidencia.
Como presidente, Obama emprendió políticas para lograr estabilizar la economía norteamericana, golpeada tras la crisis financiera de 2008, entre ellas la Ley de Reinversión y Recuperación de 2009, un paquete de estímulos económicos de alrededor de 800 000 millones de dólares.1​ Esta ley, de conjunto a las aprobadas desde la Reserva Federal así como el rescate al sector automovilístico, fueron vitales en contener la crisis. De igual manera, su Administración impulsó la reforma del sector bancario y financiero (a través de Ley Dodd-Frank),2​ la cual intentó poner freno a los excesos bancarios y una mayor protección a los consumidores frente a la crisis. A pesar de los buenos resultados económicos3​ (crecimiento del PIB sostenido de un promedio de 2.1 puntos anuales, desempleo por debajo del 5 %),4​ estos no se han revertido en la mayoría de los ciudadanos.5​ Salarios estancados y bajos índices de reducción de la pobreza, han sido elementos claves de la frustración popular hacia Obama, elementos bien utilizados por su sucesor Donald Trump en su victoria en los comicios de noviembre de 2016.6​
En política interna de la Administración Obama se destacan la Ley de Protección al Paciente y Cuidado de Salud Asequible, más conocida como Obamacare, la cual pretendía garantizar el acceso a cobertura sanitaria a cerca de 20 millones de estadounidenses. Destacado ha sido también sus políticas en materia de matrimonio igualitario7​ y a favor de la comunidad LGTBI, como por ejemplo la revocación de la política Don't ask, don't tell sobre homosexualidad en el Ejército;8​ así como su apuesta en materia de cambio climático y protección medioambiental.1​9​""",
    """Trump refused to concede after losing the 2020 presidential election to Joe Biden, falsely claiming widespread electoral fraud, and attempted to overturn the results by pressuring government officials, mounting scores of unsuccessful legal challenges, and obstructing the presidential transition. On January 6, 2021, he urged his supporters to march to the U.S. Capitol, which many of them then attacked, resulting in multiple deaths and interrupting the electoral vote count.
Trump is the only American president to have been impeached twice. After he tried to pressure Ukraine in 2019 to investigate Biden, he was impeached by the House of Representatives for abuse of power and obstruction of Congress. He was acquitted by the Senate in February 2020. The House impeached him again in January 2021 for incitement of insurrection. The Senate acquitted him in February. Scholars and historians rank Trump as one of the worst presidents in American history.[1][2]
Since leaving office, Trump has continued to dominate the Republican Party and is a candidate in the 2024 Republican presidential primaries. In 2023, a civil trial jury found that Trump sexually abused E. Jean Carroll. In 2024, a New York state court found Trump liable for financial fraud. Trump is appealing both judgments. He was also indicted in New York on 34 felony counts of falsifying business records, in Florida on 40 felony counts related to his mishandling of classified documents, in Washington, D.C., on four felony counts of conspiracy and obstruction for efforts to overturn the 2020 presidential election, and in Georgia on 13 charges of racketeering and other alleged felonies committed in an effort to overturn the state's 2020 election results. Trump pleaded not guilty to all charges""",
]


def align_time(align_fn, tokenized_set: TokenizedSet):
    ts = time()
    align_fn(tokenized_set)
    te = time()
    return te - ts


def compute_times(
    sizes: List[Tuple[int, int]],
    aligners: List[Aligner],
    tokenizers: List[PreTrainedTokenizerBase],
    reps: int,
) -> Dict:
    # Compute times
    times = {size: defaultdict(list) for size in sizes}
    for size_a, size_b in sizes:
        for _ in range(reps):
            for text in texts:
                tokenized_a = tokenizers[0](
                    [text], truncation=True, max_length=size_a
                )
                tokenized_b = tokenizers[1](
                    [text], truncation=True, max_length=size_b
                )

                tokens_a = tokenized_a.tokens()
                tokens_b = tokenized_b.tokens()

                word_ids_a = tokenized_a.word_ids()
                word_ids_b = tokenized_b.word_ids()

                tokenized_set = TokenizedSet(
                    tokens=[tokens_a, tokens_b],
                    word_ids=[word_ids_a, word_ids_b],
                    text=text,
                )
                for aligner in aligners:
                    times[(size_a, size_b)][aligner.__class__.__name__].append(
                        align_time(aligner.align, tokenized_set)
                    )

    # Aggregate statistics
    for size in times:
        for aligner in times[size]:
            times[size][aligner] = {
                "mean": np.mean(times[size][aligner]),
                "std": np.std(times[size][aligner]),
            }

    return times


def prepare_dataframe(
    times: Dict, output_file: str = "benchmark.md"
) -> pd.DataFrame:
    df_data = []
    for key, value in times.items():
        for algo, stats in value.items():
            df_data.append(
                (f"{key[0]}-{key[1]}", algo, stats["mean"], stats["std"])
            )
    df = pd.DataFrame(df_data, columns=["Tokens", "Algorithm", "Mean", "Std"])
    df.to_markdown(output_file, index=False)
    return df


def plot_times(df: pd.DataFrame, output_file: str = "benchmark.png") -> None:
    plt.figure(figsize=(10, 6))
    for algo in df["Algorithm"].unique():
        algo_df = df[df["Algorithm"] == algo]
        plt.errorbar(
            algo_df["Tokens"],
            algo_df["Mean"],
            label=algo,
            marker="o",
            linestyle="-",
        )

    plt.yscale("log")
    plt.xlabel("Token size")
    plt.ylabel("Mean time (log seconds)")
    plt.title("Benchmark")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)


if __name__ == "__main__":
    tokenizers = [
        AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1"),
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf"),
    ]
    aligners = [
        DTWAligner(distance_name="levenshtein"),
        PythonDTWAligner(distance_name="levenshtein"),
        GreedyDistanceAligner(distance_name="levenshtein"),
        PythonGreedyCoverageAligner(),
        GreedyCoverageAligner(),
        FastDTWAligner(distance_name="euclidean"),
        TamuheyAligner(),
        WordIdsAligner(),
    ]
    reps = 100
    sizes = [
        (64, 32),
        (64, 64),
        (128, 64),
        (128, 128),
        (256, 128),
        (256, 256),
        (512, 256),
        (512, 512),
    ]

    times = compute_times(sizes, aligners, tokenizers, reps)
    df = prepare_dataframe(times)
    plot_times(df)
