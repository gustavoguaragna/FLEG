"""FLEG é uma solução para reduzir impactos negativos ao treinar um modelo de classificação em ambientes com dados não-IID em Aprendizado Federado. 
Contudo, o código não simula a federação, de forma que os clientes estão serializados e o treinamento local e do servidor ocorrem na mesma máquina.
Para facilitar a compreensão, sinalizamos quando o código seria paralelizável. Além disso, indicamos quando o código se refere ao cliente ou ao servidor."""

import os
import json
import time
from collections import OrderedDict
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split, Subset
from task import (
    augment_client_with_generated,
    ClassifierHead1, ClassifierHead2, ClassifierHead3, ClassifierHead4,
    ClassifierHead1_Cifar, ClassifierHead2_Cifar, ClassifierHead3_Cifar, ClassifierHead4_Cifar,
    ClassPartitioner,
    EmbeddingGAN1, EmbeddingGAN2, EmbeddingGAN3, EmbeddingGAN4,
    EmbeddingGAN1_Cifar, EmbeddingGAN2_Cifar, EmbeddingGAN3_Cifar, EmbeddingGAN4_Cifar,
    EmbeddingPairDataset,
    FeatureExtractor1, FeatureExtractor2, FeatureExtractor3, FeatureExtractor4,
    FeatureExtractor1_Cifar, FeatureExtractor2_Cifar, FeatureExtractor3_Cifar, FeatureExtractor4_Cifar,
    GeneratedAssetDataset,
    get_label_counts,
    Net, Net_Cifar,
    unpack_batch
)
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate_inplace
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset
from tqdm import tqdm
import argparse
import random
import math

def parse_arguments():
    """Lida exclusivamente com a configuração e leitura de argumentos passados via CLI."""
    parser = argparse.ArgumentParser(description="Argumentos para o FLEG")

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--local_test_frac", type=float, default=0.2)
    parser.add_argument("--num_chunks", type=int, default=100)
    parser.add_argument("--num_partitions", type=int, default=4)
    parser.add_argument("--partitioner", type=str, choices=["ClassPartitioner", "Dirichlet"], default="ClassPartitioner")
    parser.add_argument("--strategy", type=str, choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--mu", type=float, default=0.5, help="parametro do fedprox")
    parser.add_argument("--num_syn", type=str, choices=["dynamic", "fixed"], default="dynamic")

    parser.add_argument("--beta1_disc", type=float, default=0.5)
    parser.add_argument("--beta1_gen", type=float, default=0.5)
    parser.add_argument("--beta2_disc", type=float, default=0.999)
    parser.add_argument("--beta2_gen", type=float, default=0.999)
    parser.add_argument("--lr_disc", type=float, default=0.0002)
    parser.add_argument("--lr_gen", type=float, default=0.0002)

    parser.add_argument("--checkpoint_level", type=int, default=None)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--gen_ite", type=int, default=20)
    parser.add_argument("--gan_epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10, help="Número de épocas para esperar por melhoria")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--baseline", action="store_true", help="Executa o treinamento apenas do classificador servindo como baseline")

    args = parser.parse_args()

    # Tratamento de argumentos
    if args.partitioner == "Dirichlet":
        args.partitioner = f"Dir{str(args.alpha).replace('.', '')}"

    if args.test_mode:
        print("Test Mode ativado! Reduzindo chunks, épocas da GAN e paciência.")
        args.num_chunks = 2
        args.gan_epochs = 2
        args.patience = 2

    return args

def setup_environment_and_models(args, device):
    """Configura o particionador, inicializa os modelos (Redes) e carrega checkpoints."""
    # Define o particionamento
    if args.partitioner.startswith("Dir"):
        print(f"Alpha (for Dirichlet): {args.alpha}")
        partitioner = DirichletPartitioner(
            num_partitions=args.num_partitions,
            partition_by="label",
            alpha=args.alpha,
            min_partition_size=0,
            self_balancing=False
        )
    else:
        partitioner = ClassPartitioner(num_partitions=args.num_partitions, seed=args.seed, label_column="label")

    # Inicialização dos modelos com base no dataset
    models_dict = {}
    if args.dataset == "mnist":
        image_key = "image"
        models_dict["global_net"] = Net(args.seed).to(device)
        models_dict["best_model"] = Net().to(device)
        models_dict["nets"] = [Net(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets1"] = [ClassifierHead1(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets2"] = [ClassifierHead2(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets3"] = [ClassifierHead3(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets4"] = [ClassifierHead4(args.seed).to(device) for _ in range(args.num_partitions)]
    elif args.dataset == "cifar10":
        image_key = "img"
        models_dict["global_net"] = Net_Cifar(args.seed).to(device)
        models_dict["best_model"] = Net_Cifar().to(device)
        models_dict["nets"] = [Net_Cifar(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets1"] = [ClassifierHead1_Cifar(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets2"] = [ClassifierHead2_Cifar(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets3"] = [ClassifierHead3_Cifar(args.seed).to(device) for _ in range(args.num_partitions)]
        models_dict["nets4"] = [ClassifierHead4_Cifar(args.seed).to(device) for _ in range(args.num_partitions)]
    else:
        raise ValueError("Conjunto de dados não suportado.")

    models_dict["image_key"] = image_key

    # Criação de pastas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_root = os.path.join(script_dir, "experiments")
    if args.baseline:
        experiment_name = f"{args.dataset}_{args.partitioner}_{args.strategy}_baseline_trial{args.trial}"
    else:
        experiment_name = f"{args.dataset}_{args.partitioner}_{args.strategy}_numchunks{args.num_chunks}_ganepochs{args.gan_epochs}_{args.num_syn}_fleg_trial{args.trial}"
    folder = os.path.join(experiments_root, experiment_name)
    os.makedirs(folder, exist_ok=True)

    # Carregamento do Checkpoint
    start_level = 0
    checkpoint_loaded = None
    if args.checkpoint_level is not None:
        checkpoint_path = os.path.join(folder, f"checkpoint_level{args.checkpoint_level}.pth")
        checkpoint_loaded = torch.load(checkpoint_path)
        models_dict["global_net"].load_state_dict(checkpoint_loaded['classifier_state_dict'])
        models_dict["global_net"].to(device)
        start_level = args.checkpoint_level

        # Mapeamento do checkpoint dinâmico
        head_map = {
            "mnist": {1: (ClassifierHead1, FeatureExtractor1), 2: (ClassifierHead2, FeatureExtractor2),
                      3: (ClassifierHead3, FeatureExtractor3), 4: (ClassifierHead4, FeatureExtractor4)},
            "cifar10": {1: (ClassifierHead1_Cifar, FeatureExtractor1_Cifar), 2: (ClassifierHead2_Cifar, FeatureExtractor2_Cifar),
                        3: (ClassifierHead3_Cifar, FeatureExtractor3_Cifar), 4: (ClassifierHead4_Cifar, FeatureExtractor4_Cifar)}
        }
        
        HeadClass, ExtractorClass = head_map[args.dataset][start_level]
        class_head = HeadClass(args.seed).to(device)
        feature_extractor = ExtractorClass(args.seed).to(device)
        
        class_head.load_state_dict(models_dict["global_net"].state_dict(), strict=False)
        feature_extractor.load_state_dict(models_dict["global_net"].state_dict(), strict=False)
        
        models_dict["class_head"] = class_head
        models_dict["feature_extractor"] = feature_extractor

    return partitioner, models_dict, start_level, folder, checkpoint_loaded


def prepare_datasets(args, partitioner, image_key):
    """Carrega o dataset, particiona entre os clientes, em treino e teste e divide em chunks."""
    fds = FederatedDataset(dataset=args.dataset, partitioners={"train": partitioner})
    train_partitions = [fds.load_partition(i, split="train") for i in range(args.num_partitions)]

    if args.test_mode:
        num_samples = [int(len(train_partition)/100) for train_partition in train_partitions]
        train_partitions = [train_partition.select(range(n)) for train_partition, n in zip(train_partitions, num_samples)]

    # Transformações
    if args.dataset == "mnist":
        transforms_comp = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    else:
        transforms_comp = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch[image_key] = [transforms_comp(img) for img in batch[image_key]]
        return batch

    train_partitions = [train_partition.with_transform(apply_transforms) for train_partition in train_partitions]

    testpartition = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testpartition, batch_size=64)

    client_datasets = []
    total_train_samples = 0

    # Separação de treino e teste local em cada cliente. No artigo do FLEG, não são apresentadas análises sobre os conjuntos de teste locais.
    for train_part in train_partitions:
        total = len(train_part)
        test_size = int(total * args.local_test_frac)
        train_size = total - test_size

        client_train, client_test = random_split(
            train_part, [train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        client_datasets.append({"train": client_train, "test": client_test})
        total_train_samples += train_size

    # Contagem de rótulos de cada cliente
    counts = [get_label_counts(client_data["train"]) for client_data in client_datasets]

    # Segmentação dos dados em chunks
    client_chunks = []
    for train_data in client_datasets:
        dts = train_data["train"]
        n = len(dts)
        indices = list(range(n))
        random.seed(args.seed)
        random.shuffle(indices)

        chunk_size = math.ceil(n / args.num_chunks)
        chunks = []
        for i in range(args.num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            chunk_indices = indices[start:end]
            chunks.append(Subset(dts, chunk_indices))
        client_chunks.append(chunks)

    return {
        "client_datasets": client_datasets,
        "testloader": testloader,
        "client_chunks": client_chunks,
        "counts": counts,
        "total_train_samples": total_train_samples
    }

def train_gan_for_level(args, level, models_dict, client_chunks, global_net, metrics_dict, device):
    """
    Inicializa, treina a GAN (Discriminadoras locais e Geradora global) para o nível atual 
    e gera o dataset sintético de embeddings.
    """
    dataset = args.dataset
    seed = args.seed
    num_partitions = args.num_partitions
    image_key = models_dict["image_key"]
    latent_dim = 128
    batch_size_gen = 1
    batch_size = args.batch_size
    
    # Dicionário de mapeamento para eliminar os If/Elifs
    gan_map = {
        "mnist": {
            1: (ClassifierHead1, FeatureExtractor1, EmbeddingGAN1),
            2: (ClassifierHead2, FeatureExtractor2, EmbeddingGAN2),
            3: (ClassifierHead3, FeatureExtractor3, EmbeddingGAN3),
            4: (ClassifierHead4, FeatureExtractor4, EmbeddingGAN4)
        },
        "cifar10": {
            1: (ClassifierHead1_Cifar, FeatureExtractor1_Cifar, EmbeddingGAN1_Cifar),
            2: (ClassifierHead2_Cifar, FeatureExtractor2_Cifar, EmbeddingGAN2_Cifar),
            3: (ClassifierHead3_Cifar, FeatureExtractor3_Cifar, EmbeddingGAN3_Cifar),
            4: (ClassifierHead4_Cifar, FeatureExtractor4_Cifar, EmbeddingGAN4_Cifar)
        }
    }

    # 1. Instancia as redes do nível atual
    HeadClass, ExtractorClass, GanClass = gan_map[dataset][level+1]
    
    class_head = HeadClass(seed).to(device)
    feature_extractor = ExtractorClass(seed).to(device)
    
    # 2. Transfere os pesos do modelo global para a extratora e classificadora
    pretrained_dict = global_net.state_dict()
    class_head.load_state_dict({k: v for k, v in pretrained_dict.items() if k in class_head.state_dict()})
    feature_extractor.load_state_dict({k: v for k, v in pretrained_dict.items() if k in feature_extractor.state_dict()})

    # 3. Inicializa os modelos da GAN (Clientes e Servidor)
    discs = [GanClass(condition=True, seed=seed) for _ in range(num_partitions)]
    gen = GanClass(condition=True, seed=seed).to(device)

    optim_G = torch.optim.Adam(gen.generator.parameters(), lr=args.lr_gen, betas=(args.beta1_gen, args.beta2_gen))
    optim_Ds = [torch.optim.Adam(model.discriminator.parameters(), lr=args.lr_disc, betas=(args.beta1_disc, args.beta2_disc)) for model in discs]

    # 4. Loop de Treinamento da GAN
    gan_bar = tqdm(range(args.gan_epochs), desc=f"Treinamento da GAN (Nível {level+1})", leave=True, position=1)
    
    for epoch_gan in gan_bar:
        epoch_gan_start_time = time.time()
        g_loss_c, d_loss_c, total_d_samples = 0.0, 0.0, 0
        
        chunk_bar = tqdm(range(args.num_chunks), desc="Chunks", leave=True, position=2)

        for chunk_idx in chunk_bar:
            chunk_start_time = time.time()
            d_loss_b, total_chunk_samples = 0, 0
            clients_disc_time = []

            # 4.1 Treinamento das Discriminadoras (Clientes)
            for cliente, (disc, chunks) in enumerate(zip(discs, client_chunks)):
                chunk_dataset = chunks[chunk_idx]
                if len(chunk_dataset) == 0: continue

                chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)
                disc.to(device)
                optim_D = optim_Ds[cliente]

                start_disc_time = time.time()
                for batch in chunk_loader:
                    images, labels = batch[image_key].to(device), batch["label"].to(device)
                    if images.size(0) == 1: continue

                    with torch.no_grad():
                        images = feature_extractor(images)

                    real_ident = torch.full((images.size(0), 1), 1., device=device)
                    fake_ident = torch.full((images.size(0), 1), 0., device=device)
                    z_noise = torch.randn(images.size(0), latent_dim, device=device)
                    x_fake_labels = torch.randint(0, 10, (images.size(0),), device=device)

                    optim_D.zero_grad()

                    y_real = disc(images, labels)
                    d_real_loss = disc.loss(y_real, real_ident)
                    
                    x_fake = gen(z_noise, x_fake_labels).detach() # Em teoria, x_fake, assim como z_noise e x_fake_labels, é fabricado no servidor e enviado para os clientes, a fim de diminuir o custo de transferir a geradora.
                    y_fake_d = disc(x_fake, x_fake_labels)
                    d_fake_loss = disc.loss(y_fake_d, fake_ident)

                    d_loss = (d_real_loss + d_fake_loss) / 2
                    d_loss.backward()
                    optim_D.step()
                    
                    d_loss_b += d_loss.item()
                    total_chunk_samples += images.size(0)
                    
                clients_disc_time.append(time.time() - start_disc_time)

            # Atualiza métricas do Discriminador
            avg_d_loss_chunk = d_loss_b / total_chunk_samples if total_chunk_samples > 0 else 0.0
            metrics_dict["d_losses_chunk"].append(avg_d_loss_chunk)
            d_loss_c += avg_d_loss_chunk * total_chunk_samples
            total_d_samples += total_chunk_samples
            if clients_disc_time: metrics_dict["disc_time"].append(sum(clients_disc_time) / len(clients_disc_time))

            # 4.2 Treinamento da Geradora (Servidor)
            chunk_g_loss = 0.0
            start_gen_time = time.time()
            for _ in range(args.gen_ite):
                optim_G.zero_grad()
                z_noise = torch.randn(batch_size_gen, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size_gen,), device=device)
                x_fake = gen(z_noise, x_fake_labels)

                # Estratégia Dmax
                y_fake_gs = [model(x_fake.detach(), x_fake_labels) for model in discs]
                y_fake_g_means = [torch.mean(y).item() for y in y_fake_gs]
                Dmax = discs[y_fake_g_means.index(max(y_fake_g_means))]

                real_ident = torch.full((batch_size_gen, 1), 1., device=device)
                y_fake_g = Dmax(x_fake, x_fake_labels)
                g_loss = gen.loss(y_fake_g, real_ident)

                g_loss.backward()
                optim_G.step()
                gen.to(device)
                chunk_g_loss += g_loss.item()

            # Atualiza métricas do Gerador
            metrics_dict["g_losses_chunk"].append(chunk_g_loss / args.gen_ite)
            g_loss_c += chunk_g_loss / args.gen_ite
            metrics_dict["gen_time"].append(time.time() - start_gen_time)
            metrics_dict["time_chunk"].append(time.time() - chunk_start_time)

        # Atualiza métricas da Época da GAN
        g_loss_e = g_loss_c / args.num_chunks
        d_loss_e = d_loss_c / total_d_samples if total_d_samples > 0 else 0.0
        metrics_dict["g_losses_epoch"].append(g_loss_e)
        metrics_dict["d_losses_epoch"].append(d_loss_e)
        metrics_dict["time_epoch_gan"].append(time.time() - epoch_gan_start_time)

    # 5. Geração do Dataset Sintético pelo Servidor
    start_img_syn_time = time.time()
    
    # Substituí a variável global 'D' calculando novamente a partir dos chunks para evitar vazamento de escopo
    total_samples = sum([len(c) for chunks in client_chunks for c in chunks]) 
    
    num_samples = int(math.ceil(total_samples/num_partitions)*(level+1)/4) if args.num_syn == "dynamic" else (48000 if dataset == "mnist" else 40000)
    
    generated_dataset = GeneratedAssetDataset(
        generator=gen, 
        num_samples=num_samples, 
        latent_dim=latent_dim, 
        num_classes=10, 
        asset_shape=(gen.embedding_dim,),
        asset_col_name=image_key,
        device=device
    )
    metrics_dict["img_syn_time"].append(time.time() - start_img_syn_time)
    
    # Retorna o que precisamos manter no fluxo principal (A geradora, o dataset e a extratora/classificadora se houver necessidade)
    return generated_dataset, class_head, feature_extractor

def run_training_loop(args, models_dict, data_dict, start_level, folder, checkpoint_loaded, device):
    """Executa o loop de treinamento principal das redes classificadoras e das GANs."""
    # Desempacota variáveis úteis
    dataset, num_partitions, num_chunks, strategy, mu = args.dataset, args.num_partitions, args.num_chunks, args.strategy, args.mu
    batch_size, patience, baseline, levels = args.batch_size, args.patience, args.baseline, args.levels
    gan_epochs, lr_gen, lr_disc = args.gan_epochs, args.lr_gen, args.lr_disc
    
    global_net, best_model = models_dict["global_net"], models_dict["best_model"]
    image_key = models_dict["image_key"]
    
    client_datasets, testloader = data_dict["client_datasets"], data_dict["testloader"]
    client_chunks, counts = data_dict["client_chunks"], data_dict["counts"]
    D = data_dict["total_train_samples"]

    criterion = torch.nn.CrossEntropyLoss()
    metrics_filename = f"{folder}/metrics.json"

    # Inicializa métricas 
    metrics_dict = {
                "g_losses_chunk": [],
                "d_losses_chunk": [],
                "g_losses_epoch": [],
                "d_losses_epoch": [],
                "net_loss": [],
                "net_acc": [],
                "time_epoch_classifier": [],
                "time_epoch_gan": [],
                "time_chunk": [],
                "time_level": [],
                "net_time": [],
                "net_global_eval_time": [],
                "disc_time": [],
                "gen_time": [],
                "img_syn_time": [],
                "MB_transmission": [],
                "traffic_cost_classifier": [],
                "traffic_cost_discriminator": [],
                "accuracy_transition": [],
                "x_fake_samples_size": [],
                "syn_emb_ds": []
            }
    
    # Tamanhos dos modelos de cada nível em MB para cálculo do custo de comunicação.
    SIZE_CLASSIFIER = {
    'cifar10': {1: 0.25, 2: 0.248, 3: 0.238, 4: 0.045, 5: 0.004},
    'mnist': {1: 0.18, 2: 0.179, 3: 0.169, 4: 0.045, 5: 0.004}
    }
    SIZE_DISC = {
        'cifar10': {1: 18.12, 2: 3.79, 3: 0.79, 4: 0.23},
        'mnist': {1: 5.69, 2: 1.08, 3: 0.8, 4: 0.23}
    }

    # Tamanho dos embeddings compartilhados para treinamento da discriminadora e classificadora
    emb_samples_disc = {
        'mnist':   {0: [41.568], 1: [12.384], 2: [5.856], 3: [4.128]}, 
        'cifar10': {0: [47.12],  1: [16.08],  2: [4.88],  3: [3.44]}
    }  
    emb_samples_classifier = {
        'mnist': {
            'dynamic': {0: [10.392],  1: [6.192],  2: [4.392],  3: [4.128]},  
            'fixed':   {0: [166.272], 1: [49.536], 2: [23.424], 3: [16.512]}
        }, 
        'cifar10': {
            'dynamic': {0: [11.78],   1: [8.04],  2: [3.66],  3: [3.44]},  
            'fixed':   {0: [188.48],  1: [64.32], 2: [19.52], 3: [13.76]}
        }
    }

    batch_size_gen = 1
    latent_dim = 128
    best_accuracy = 0

    level_bar = tqdm(range(start_level, levels+1), desc="Training", leave=True, position=0)

    for level in level_bar:
        epochs_this_level = 0
        level_start_time = time.time()
        epochs_no_improve = 0
        
        # Recupera as redes do nível atual e extrai as class heads (caso não existam no models_dict ainda)
        if level == 0:
            nets_level = models_dict["nets"]
        else:
            nets_level = models_dict[f"nets{level}"]
            class_head = models_dict.get("class_head")
            feature_extractor = models_dict.get("feature_extractor")

        optims = [torch.optim.Adam(net.parameters(), lr=0.01) for net in nets_level]
        
        # --- TREINAMENTO DO CLASSIFICADOR ---
        client_combined_loaders = {}
        if level > 0:
            print("Adicionando datasets sintéticos para todos os clientes...")
            if checkpoint_loaded is not None:
                generated_dataset = checkpoint_loaded['generated_dataset']
            
            for cliente in range(num_partitions):
                trainloader = DataLoader(client_datasets[cliente]["train"], batch_size=batch_size, shuffle=False)
                all_embeddings, all_labels = [], []
                with torch.no_grad():
                    feature_extractor.eval()
                    for batch in trainloader:
                        images, labels = unpack_batch(batch, image_key=image_key, label_key='label')
                        embeddings = feature_extractor(images.to(device)).view(images.size(0), -1)
                        all_embeddings.append(embeddings)
                        all_labels.append(labels.to(device))
                
                final_embeddings = torch.cat(all_embeddings, dim=0)
                final_labels = torch.cat(all_labels, dim=0)
                embedding_dataset = EmbeddingPairDataset(final_embeddings, final_labels,
                                                asset_col_name=generated_dataset.asset_col_name,
                                                label_col_name=generated_dataset.label_col_name)
                combined_ds, stats = augment_client_with_generated(
                    client_train=embedding_dataset,
                    gen_dataset=generated_dataset,
                    counts=counts[cliente],    
                    strategy='threshold', 
                    fill_to=int(len(embedding_dataset)/10),
                    threshold=int(len(embedding_dataset)/10),
                    rng_seed=42
                )
                print(f"Cliente {cliente}: adicionou {stats['gen_selected_count']} amostras geradas para as classes {stats['desired_labels']}")
                client_combined_loaders[cliente] = DataLoader(combined_ds, batch_size=batch_size, shuffle=True)

        # Loop de treinamento do classificador
        while epochs_no_improve <= patience:
            epoch_start_time = time.time()
            params, results, net_times, running_losses = [], [], [], [0.0]*num_partitions

            client_bar = tqdm(enumerate(nets_level), desc="Clients", leave=True, position=1)
            for cliente, net in client_bar:
                net.load_state_dict(global_net.state_dict() if level == 0 else class_head.state_dict(), strict=True)
                optim = optims[cliente]
                net.train()

                if level == 0:
                    combined_dataloader = DataLoader(client_datasets[cliente]["train"], batch_size=batch_size, shuffle=True)
                else:
                    combined_dataloader = client_combined_loaders[cliente]

                start_net_time = time.time()
                
                for batch in combined_dataloader:
                    images, labels = batch[image_key].to(device), batch["label"].to(device)
                    if images.size(0) == 1: continue
                    optim.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    # FedProx simplificado
                    if strategy == "fedprox":
                        proximal_term = sum((loc - glob).norm(2) for loc, glob in zip(net.parameters(), global_net.parameters() if level == 0 else class_head.parameters()))
                        loss += (mu / 2) * proximal_term
                    loss.backward()
                    optim.step()
                    running_losses[cliente] += loss.item()

                net_times.append(time.time() - start_net_time)
                running_losses[cliente] /= len(combined_dataloader)

                params.append(ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]))
                results.append((cliente, FitRes(status=Status(code=Code.OK, message="Success"), parameters=params[cliente], num_examples=len(client_datasets[cliente]["train"]), metrics={})))

            metrics_dict["net_time"].append(sum(net_times) / len(net_times))
            metrics_dict["net_loss"].append(sum(running_losses) / len(running_losses))
            
            # Agregação do servidor
            aggregated_ndarrays = aggregate_inplace(results)
            if level == 0:
                params_dict = zip(global_net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
                global_net.load_state_dict(state_dict, strict=True)
            else:
                params_dict = zip(class_head.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
                class_head.load_state_dict(state_dict, strict=True)

            # Avaliação do modelo global no conjunto de validação no servidor
            correct, loss = 0, 0.0
            net_global_eval_start_time = time.time()
            with torch.no_grad():
                for batch in testloader:
                    images = batch[image_key].to(device)
                    labels = batch["label"].to(device)
                    if level == 0:
                        outputs = global_net(images)
                    else:
                        outputs = class_head(feature_extractor(images))
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
            accuracy = correct / len(testloader.dataset)
            metrics_dict["net_global_eval_time"].append(time.time() - net_global_eval_start_time)
            metrics_dict["net_acc"].append(accuracy)

            metrics_dict["time_epoch_classifier"].append(time.time() - epoch_start_time)
            epochs_this_level += 1

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_no_improve = 0
                if level == 0:
                    best_model.load_state_dict(global_net.state_dict())
                else:
                    best_model.load_state_dict(class_head.state_dict(), strict=False)
            else:
                epochs_no_improve += 1
                print(f"Sem melhorias por {epochs_no_improve} épocas. Melhor acurácia: {best_accuracy:.4f}")

        global_net.load_state_dict(best_model.state_dict())

        # Salva métricas e finaliza o treino caso seja baseline
        if baseline:
            metrics_dict["time_level"].append(time.time() - level_start_time)
            try:
                with open(metrics_filename, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
                print(f"Dicionário de métricas salvo em {metrics_filename}")
            except Exception as e:
                print(f"Erro ao salvar o dicionário de métricas para JSON: {e}")
            
            checkpoint = {
                    'classifier_state_dict': global_net.state_dict(),
                }
            checkpoint_file = f"{folder}/checkpoint.pth"
            torch.save(checkpoint, checkpoint_file)
            print("baseline trained")
            break

        class_up_down_MB = SIZE_CLASSIFIER[dataset][level+1] * epochs_this_level
        metrics_dict["traffic_cost_classifier"].append(class_up_down_MB)

        metrics_dict["accuracy_transition"].append(accuracy)

        # --- TREINAMENTO DA GAN ---
        print(f"Alternando para o nível {level} de treinamento da GAN.")
        
        # Se for o último nível, ele deve pular a GAN e encerrar (como estava no seu if original "else: break")
        if level+1 > levels:
            metrics_dict["time_level"].append(time.time() - level_start_time)
            metrics_dict["MB_transmission"].append(class_up_down_MB)

            try:
                with open(metrics_filename, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
                print(f"Dicionário de métricas salvo em {metrics_filename}")
            except Exception as e:
                print(f"Erro ao salvar o dicionário de métricas para JSON: {e}")

            checkpoint = {'classifier_state_dict': global_net.state_dict(), 'level': level+1}
            torch.save(checkpoint, f"{folder}/checkpoint_end.pth")
            print("Máximo de níveis atingido. Treinamento finalizado!")
            break
            
        # Chama a nossa nova função super limpa!
        generated_dataset, class_head, feature_extractor = train_gan_for_level(
            args=args, 
            level=level, 
            models_dict=models_dict, 
            client_chunks=client_chunks, 
            global_net=global_net, 
            metrics_dict=metrics_dict, 
            device=device
        )
        
        # Atualiza o models_dict com as instâncias criadas na GAN para o próximo nível usar
        models_dict["class_head"] = class_head
        models_dict["feature_extractor"] = feature_extractor
        checkpoint_loaded = {'generated_dataset': generated_dataset} # Finge o comportamento do checkpoint para o loop da proxima epoch

        # Computa métricas finais do nível (custo de rede, etc)
        gan_up_MB = SIZE_DISC[dataset][level+1] * gan_epochs * num_chunks
        metrics_dict["traffic_cost_discriminator"].append(gan_up_MB)

        metrics_dict["x_fake_samples_size"].append(emb_samples_disc[dataset][level][0])
        metrics_dict["syn_emb_ds"].append(emb_samples_classifier[dataset][args.num_syn][level][0])
        metrics_dict["MB_transmission"].append(class_up_down_MB + gan_up_MB  + emb_samples_disc[dataset][level][0] + emb_samples_classifier[dataset][args.num_syn][level][0])
        metrics_dict["time_level"].append(time.time() - level_start_time)

        # Salva as métricas no final do loop
        with open(metrics_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
            
        # Salva o checkpoint com o dataset recém gerado
        checkpoint = {
                'classifier_state_dict': global_net.state_dict(),
                'level': level+1,
                'generated_dataset': generated_dataset,
            }
        checkpoint_file = f"{folder}/checkpoint_level{level+1}.pth"
        torch.save(checkpoint, checkpoint_file)

        print(f"Nível {level+1} completo")

# --- BLOCO DE ORQUESTRAÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # 1. Obter argumentos
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"""FLEG Configuration:
    Checkpoint Level: {args.checkpoint_level}
    Dataset: {args.dataset}
    Num Chunks: {args.num_chunks}
    Num Partitions: {args.num_partitions}
    Partitioner: {args.partitioner}
    Strategy: {args.strategy}
    Device: {device}
    """)

    # 2. Configurar o ambiente e inicializar redes/checkpoints
    partitioner, models_dict, start_level, folder, checkpoint_loaded = setup_environment_and_models(args, device)
    
    # 3. Preparar os dados (Treino, Teste e Divisões)
    data_dict = prepare_datasets(args, partitioner, models_dict["image_key"])
    
    # 4. Executar o Loop Principal
    run_training_loop(args, models_dict, data_dict, start_level, folder, checkpoint_loaded, device)
