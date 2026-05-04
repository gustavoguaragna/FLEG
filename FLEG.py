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
        print("Modo de teste ativado! Reduzindo chunks, épocas da GAN e paciência.")
        args.num_chunks = 2
        args.gan_epochs = 2
        args.patience = 2

    return args

def setup_environment_and_models(args, device):
    """Configura o particionador, inicializa os modelos (Redes) e carrega checkpoints."""
    # Define o particionamento
    if args.partitioner.startswith("Dir"):
        print(f"Alpha para Dirichlet: {args.alpha}")
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
    experiment_dir = os.path.join(experiments_root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Carregamento do Checkpoint
    start_level = 0
    resume_checkpoint = None
    if args.checkpoint_level is not None:
        checkpoint_path = os.path.join(experiment_dir, f"checkpoint_level{args.checkpoint_level}.pth")
        resume_checkpoint = torch.load(checkpoint_path)
        models_dict["global_net"].load_state_dict(resume_checkpoint['classifier_state_dict'])
        models_dict["global_net"].to(device)
        start_level = args.checkpoint_level

        # Mapeamento do checkpoint dinâmico
        head_map = {
            "mnist": {1: (ClassifierHead1, FeatureExtractor1), 2: (ClassifierHead2, FeatureExtractor2),
                      3: (ClassifierHead3, FeatureExtractor3), 4: (ClassifierHead4, FeatureExtractor4)},
            "cifar10": {1: (ClassifierHead1_Cifar, FeatureExtractor1_Cifar), 2: (ClassifierHead2_Cifar, FeatureExtractor2_Cifar),
                        3: (ClassifierHead3_Cifar, FeatureExtractor3_Cifar), 4: (ClassifierHead4_Cifar, FeatureExtractor4_Cifar)}
        }
        
        ClassifierHeadClass, FeatureExtractorClass = head_map[args.dataset][start_level]
        classifier_head = ClassifierHeadClass(args.seed).to(device)
        feature_extractor = FeatureExtractorClass(args.seed).to(device)
        
        classifier_head.load_state_dict(models_dict["global_net"].state_dict(), strict=False)
        feature_extractor.load_state_dict(models_dict["global_net"].state_dict(), strict=False)
        
        models_dict["classifier_head"] = classifier_head
        models_dict["feature_extractor"] = feature_extractor

    return partitioner, models_dict, start_level, experiment_dir, resume_checkpoint


def prepare_datasets(args, partitioner, image_key):
    """Carrega o dataset, particiona entre os clientes, em treino e teste e divide em chunks."""
    federated_dataset = FederatedDataset(dataset=args.dataset, partitioners={"train": partitioner})
    train_partitions = [federated_dataset.load_partition(i, split="train") for i in range(args.num_partitions)]

    if args.test_mode:
        test_mode_sample_counts = [int(len(train_partition)/100) for train_partition in train_partitions]
        train_partitions = [train_partition.select(range(n)) for train_partition, n in zip(train_partitions, test_mode_sample_counts)]

    # Transformações
    if args.dataset == "mnist":
        transform_pipeline = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    else:
        transform_pipeline = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch[image_key] = [transform_pipeline(img) for img in batch[image_key]]
        return batch

    train_partitions = [train_partition.with_transform(apply_transforms) for train_partition in train_partitions]

    test_partition = federated_dataset.load_split("test").with_transform(apply_transforms)
    test_loader = DataLoader(test_partition, batch_size=64)

    client_datasets = []

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

    # Contagem de rótulos de cada cliente
    client_label_counts = [get_label_counts(client_data["train"]) for client_data in client_datasets]

    # Segmentação dos dados em chunks
    client_chunks = []
    for train_data in client_datasets:
        client_train_subset = train_data["train"]
        num_client_samples = len(client_train_subset)
        indices = list(range(num_client_samples))
        random.seed(args.seed)
        random.shuffle(indices)

        chunk_size = math.ceil(num_client_samples / args.num_chunks)
        chunks = []
        for i in range(args.num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_client_samples)
            chunk_indices = indices[start:end]
            chunks.append(Subset(client_train_subset, chunk_indices))
        client_chunks.append(chunks)

    return {
        "client_datasets": client_datasets,
        "test_loader": test_loader,
        "client_chunks": client_chunks,
        "client_label_counts": client_label_counts
    }

def train_gan_for_level(args, level, models_dict, client_chunks, global_classifier, metrics_dict, device):
    """
    Inicializa, treina a GAN (Discriminadoras locais e Geradora global) para o nível atual 
    e gera o dataset sintético de embeddings.
    """
    dataset = args.dataset
    seed = args.seed
    num_partitions = args.num_partitions
    image_key = models_dict["image_key"]
    latent_dim = 128
    generator_batch_size = 1
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
    ClassifierHeadClass, FeatureExtractorClass, GanModelClass = gan_map[dataset][level+1]
    
    classifier_head = ClassifierHeadClass(seed).to(device)
    feature_extractor = FeatureExtractorClass(seed).to(device)
    
    # 2. Transfere os pesos do modelo global para a extratora e classificadora
    global_classifier_state = global_classifier.state_dict()
    classifier_head.load_state_dict({k: v for k, v in global_classifier_state.items() if k in classifier_head.state_dict()})
    feature_extractor.load_state_dict({k: v for k, v in global_classifier_state.items() if k in feature_extractor.state_dict()})

    # 3. Inicializa os modelos da GAN (Clientes e Servidor)
    client_discriminators = [GanModelClass(condition=True, seed=seed) for _ in range(num_partitions)]
    server_generator = GanModelClass(condition=True, seed=seed).to(device)

    generator_optimizer = torch.optim.Adam(server_generator.generator.parameters(), lr=args.lr_gen, betas=(args.beta1_gen, args.beta2_gen))
    discriminator_optimizers = [
        torch.optim.Adam(model.discriminator.parameters(), lr=args.lr_disc, betas=(args.beta1_disc, args.beta2_disc))
        for model in client_discriminators
    ]

    # 4. Loop de Treinamento da GAN
    gan_epoch_bar = tqdm(range(args.gan_epochs), desc=f"Treinamento da GAN (Nível {level+1})", leave=True, position=1)
    
    for _ in gan_epoch_bar:
        gan_epoch_start_time = time.time()
        generator_loss_sum, weighted_discriminator_loss_sum, total_discriminator_samples = 0.0, 0.0, 0
        
        chunk_progress_bar = tqdm(range(args.num_chunks), desc="Chunks", leave=True, position=2)

        for chunk_idx in chunk_progress_bar:
            chunk_start_time = time.time()
            chunk_discriminator_batch_loss_sum, chunk_sample_count = 0, 0
            client_discriminator_times = []

            # 4.1 Treinamento das Discriminadoras (Clientes)
            for client_idx, (client_discriminator, chunks) in enumerate(zip(client_discriminators, client_chunks)):
                chunk_dataset = chunks[chunk_idx]
                if len(chunk_dataset) == 0: continue

                chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)
                client_discriminator.to(device)
                discriminator_optimizer = discriminator_optimizers[client_idx]

                discriminator_start_time = time.time()
                for batch in chunk_loader:
                    images, labels = batch[image_key].to(device), batch["label"].to(device)
                    if images.size(0) == 1: continue

                    with torch.no_grad():
                        images = feature_extractor(images)

                    real_targets = torch.full((images.size(0), 1), 1., device=device)
                    fake_targets = torch.full((images.size(0), 1), 0., device=device)
                    noise = torch.randn(images.size(0), latent_dim, device=device)
                    fake_labels = torch.randint(0, 10, (images.size(0),), device=device)

                    discriminator_optimizer.zero_grad()

                    real_predictions = client_discriminator(images, labels)
                    discriminator_real_loss = client_discriminator.loss(real_predictions, real_targets)
                    
                    # Estes tensores seriam fabricados no servidor e enviados aos clientes.
                    fake_embeddings = server_generator(noise, fake_labels).detach()
                    fake_predictions_for_discriminator = client_discriminator(fake_embeddings, fake_labels)
                    discriminator_fake_loss = client_discriminator.loss(fake_predictions_for_discriminator, fake_targets)

                    discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
                    
                    chunk_discriminator_batch_loss_sum += discriminator_loss.item()
                    chunk_sample_count += images.size(0)
                    
                client_discriminator_times.append(time.time() - discriminator_start_time)

            # Atualiza métricas do Discriminador
            avg_discriminator_loss_chunk = chunk_discriminator_batch_loss_sum / chunk_sample_count if chunk_sample_count > 0 else 0.0
            metrics_dict["d_losses_chunk"].append(avg_discriminator_loss_chunk)
            weighted_discriminator_loss_sum += avg_discriminator_loss_chunk * chunk_sample_count
            total_discriminator_samples += chunk_sample_count
            if client_discriminator_times:
                metrics_dict["disc_time"].append(sum(client_discriminator_times) / len(client_discriminator_times))

            # 4.2 Treinamento da Geradora (Servidor)
            chunk_generator_loss = 0.0
            generator_start_time = time.time()
            for _ in range(args.gen_ite):
                generator_optimizer.zero_grad()
                noise = torch.randn(generator_batch_size, latent_dim, device=device)
                fake_labels = torch.randint(0, 10, (generator_batch_size,), device=device)
                fake_embeddings = server_generator(noise, fake_labels)

            # Estratégia Dmax
                fake_predictions_by_discriminator = [
                    model(fake_embeddings.detach(), fake_labels)
                    for model in client_discriminators
                ]
                fake_score_means = [torch.mean(predictions).item() for predictions in fake_predictions_by_discriminator]
                best_discriminator = client_discriminators[fake_score_means.index(max(fake_score_means))]

                real_targets = torch.full((generator_batch_size, 1), 1., device=device)
                fake_predictions_for_generator = best_discriminator(fake_embeddings, fake_labels)
                generator_loss = server_generator.loss(fake_predictions_for_generator, real_targets)

                generator_loss.backward()
                generator_optimizer.step()
                server_generator.to(device)
                chunk_generator_loss += generator_loss.item()

            # Atualiza métricas do Gerador
            metrics_dict["g_losses_chunk"].append(chunk_generator_loss / args.gen_ite)
            generator_loss_sum += chunk_generator_loss / args.gen_ite
            metrics_dict["gen_time"].append(time.time() - generator_start_time)
            metrics_dict["time_chunk"].append(time.time() - chunk_start_time)

        # Atualiza métricas da Época da GAN
        epoch_generator_loss = generator_loss_sum / args.num_chunks
        epoch_discriminator_loss = weighted_discriminator_loss_sum / total_discriminator_samples if total_discriminator_samples > 0 else 0.0
        metrics_dict["g_losses_epoch"].append(epoch_generator_loss)
        metrics_dict["d_losses_epoch"].append(epoch_discriminator_loss)
        metrics_dict["time_epoch_gan"].append(time.time() - gan_epoch_start_time)

    # 5. Geração do Dataset Sintético pelo Servidor
    synthetic_data_start_time = time.time()
    
    total_client_train_samples = sum(len(chunk) for chunks in client_chunks for chunk in chunks)
    
    num_synthetic_samples = int(math.ceil(total_client_train_samples/num_partitions)*(level+1)/4) if args.num_syn == "dynamic" else (48000 if dataset == "mnist" else 40000)
    
    synthetic_dataset = GeneratedAssetDataset(
        generator=server_generator, 
        num_samples=num_synthetic_samples, 
        latent_dim=latent_dim, 
        num_classes=10, 
        asset_shape=(server_generator.embedding_dim,),
        asset_col_name=image_key,
        device=device
    )
    metrics_dict["img_syn_time"].append(time.time() - synthetic_data_start_time)
    
    # Retorna o estado necessário para o próximo nível do classificador.
    return synthetic_dataset, classifier_head, feature_extractor

def run_training_loop(args, models_dict, data_dict, start_level, experiment_dir, synthetic_dataset_state, device):
    """Executa o loop de treinamento principal das redes classificadoras e das GANs."""
    # Desempacota variáveis úteis
    dataset, num_partitions, num_chunks, strategy, mu = args.dataset, args.num_partitions, args.num_chunks, args.strategy, args.mu
    batch_size, patience, baseline, levels = args.batch_size, args.patience, args.baseline, args.levels
    gan_epochs = args.gan_epochs
    
    global_classifier, best_classifier = models_dict["global_net"], models_dict["best_model"]
    image_key = models_dict["image_key"]
    
    client_datasets, test_loader = data_dict["client_datasets"], data_dict["test_loader"]
    client_chunks, client_label_counts = data_dict["client_chunks"], data_dict["client_label_counts"]

    criterion = torch.nn.CrossEntropyLoss()
    metrics_path = f"{experiment_dir}/metrics.json"

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
    CLASSIFIER_MODEL_SIZE_MB = {
    'cifar10': {1: 0.25, 2: 0.248, 3: 0.238, 4: 0.045, 5: 0.004},
    'mnist': {1: 0.18, 2: 0.179, 3: 0.169, 4: 0.045, 5: 0.004}
    }
    DISCRIMINATOR_MODEL_SIZE_MB = {
        'cifar10': {1: 18.12, 2: 3.79, 3: 0.79, 4: 0.23},
        'mnist': {1: 5.69, 2: 1.08, 3: 0.8, 4: 0.23}
    }

    # Tamanho dos embeddings compartilhados para treinamento da discriminadora e classificadora
    DISCRIMINATOR_EMBEDDING_TRANSFER_MB = {
        'mnist':   {0: [41.568], 1: [12.384], 2: [5.856], 3: [4.128]}, 
        'cifar10': {0: [47.12],  1: [16.08],  2: [4.88],  3: [3.44]}
    }  
    CLASSIFIER_SYNTHETIC_EMBEDDING_TRANSFER_MB = {
        'mnist': {
            'dynamic': {0: [10.392],  1: [6.192],  2: [4.392],  3: [4.128]},  
            'fixed':   {0: [166.272], 1: [49.536], 2: [23.424], 3: [16.512]}
        }, 
        'cifar10': {
            'dynamic': {0: [11.78],   1: [8.04],  2: [3.66],  3: [3.44]},  
            'fixed':   {0: [188.48],  1: [64.32], 2: [19.52], 3: [13.76]}
        }
    }

    best_accuracy = 0

    training_level_bar = tqdm(range(start_level, levels+1), desc="Treinamento", leave=True, position=0)

    for level in training_level_bar:
        classifier_epochs_this_level = 0
        level_start_time = time.time()
        epochs_no_improve = 0
        
        # Recupera as redes do nível atual e extrai as class heads (caso não existam no models_dict ainda)
        if level == 0:
            client_classifiers_for_level = models_dict["nets"]
        else:
            client_classifiers_for_level = models_dict[f"nets{level}"]
            classifier_head = models_dict.get("classifier_head")
            feature_extractor = models_dict.get("feature_extractor")

        client_optimizers = [torch.optim.Adam(client_classifier.parameters(), lr=0.01) for client_classifier in client_classifiers_for_level]
        
        # --- TREINAMENTO DO CLASSIFICADOR ---
        augmented_client_loaders = {}
        if level > 0:
            print("Adicionando conjuntos sintéticos para todos os clientes...")
            if synthetic_dataset_state is not None:
                synthetic_dataset = synthetic_dataset_state['generated_dataset']
            
            for client_idx in range(num_partitions):
                client_train_loader = DataLoader(client_datasets[client_idx]["train"], batch_size=batch_size, shuffle=False)
                all_embeddings, all_labels = [], []
                with torch.no_grad():
                    feature_extractor.eval()
                    for batch in client_train_loader:
                        images, labels = unpack_batch(batch, image_key=image_key, label_key='label')
                        embeddings = feature_extractor(images.to(device)).view(images.size(0), -1)
                        all_embeddings.append(embeddings)
                        all_labels.append(labels.to(device))
                
                client_embeddings = torch.cat(all_embeddings, dim=0)
                client_labels = torch.cat(all_labels, dim=0)
                client_embedding_dataset = EmbeddingPairDataset(client_embeddings, client_labels,
                                                asset_col_name=synthetic_dataset.asset_col_name,
                                                label_col_name=synthetic_dataset.label_col_name)
                augmented_dataset, augmentation_stats = augment_client_with_generated(
                    client_train=client_embedding_dataset,
                    gen_dataset=synthetic_dataset,
                    counts=client_label_counts[client_idx],    
                    strategy='threshold', 
                    fill_to=int(len(client_embedding_dataset)/10),
                    threshold=int(len(client_embedding_dataset)/10),
                    rng_seed=42
                )
                print(f"Cliente {client_idx}: adicionou {augmentation_stats['gen_selected_count']} amostras geradas para as classes {augmentation_stats['desired_labels']}")
                augmented_client_loaders[client_idx] = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

        # Loop de treinamento do classificador
        while epochs_no_improve <= patience:
            epoch_start_time = time.time()
            fit_results, client_train_times, client_epoch_losses = [], [], [0.0]*num_partitions

            client_bar = tqdm(enumerate(client_classifiers_for_level), desc="Clients", leave=True, position=1)
            for client_idx, client_classifier in client_bar:
                client_classifier.load_state_dict(global_classifier.state_dict() if level == 0 else classifier_head.state_dict(), strict=True)
                client_optimizer = client_optimizers[client_idx]
                client_classifier.train()

                if level == 0:
                    client_training_loader = DataLoader(client_datasets[client_idx]["train"], batch_size=batch_size, shuffle=True)
                else:
                    client_training_loader = augmented_client_loaders[client_idx]

                client_train_start_time = time.time()
                
                for batch in client_training_loader:
                    images, labels = batch[image_key].to(device), batch["label"].to(device)
                    if images.size(0) == 1: continue
                    client_optimizer.zero_grad()
                    logits = client_classifier(images)
                    classification_loss = criterion(logits, labels)
                    # FedProx simplificado
                    if strategy == "fedprox":
                        reference_parameters = global_classifier.parameters() if level == 0 else classifier_head.parameters()
                        proximal_term = sum(
                            (local_param - reference_param).norm(2)
                            for local_param, reference_param in zip(client_classifier.parameters(), reference_parameters)
                        )
                        classification_loss += (mu / 2) * proximal_term
                    classification_loss.backward()
                    client_optimizer.step()
                    client_epoch_losses[client_idx] += classification_loss.item()

                client_train_times.append(time.time() - client_train_start_time)
                client_epoch_losses[client_idx] /= len(client_training_loader)

                client_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in client_classifier.state_dict().items()])
                fit_results.append((client_idx, FitRes(status=Status(code=Code.OK, message="Success"), parameters=client_parameters, num_examples=len(client_datasets[client_idx]["train"]), metrics={})))

            metrics_dict["net_time"].append(sum(client_train_times) / len(client_train_times))
            metrics_dict["net_loss"].append(sum(client_epoch_losses) / len(client_epoch_losses))
            
            # Agregação do servidor
            aggregated_ndarrays = aggregate_inplace(fit_results)
            if level == 0:
                aggregated_state_items = zip(global_classifier.state_dict().keys(), aggregated_ndarrays)
                aggregated_state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in aggregated_state_items})
                global_classifier.load_state_dict(aggregated_state_dict, strict=True)
            else:
                aggregated_state_items = zip(classifier_head.state_dict().keys(), aggregated_ndarrays)
                aggregated_state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in aggregated_state_items})
                classifier_head.load_state_dict(aggregated_state_dict, strict=True)

            # Avaliação do modelo global no conjunto de validação no servidor
            correct_predictions = 0
            net_global_eval_start_time = time.time()
            with torch.no_grad():
                for batch in test_loader:
                    images = batch[image_key].to(device)
                    labels = batch["label"].to(device)
                    if level == 0:
                        logits = global_classifier(images)
                    else:
                        logits = classifier_head(feature_extractor(images))
                    correct_predictions += (torch.max(logits.data, 1)[1] == labels).sum().item()
            
            accuracy = correct_predictions / len(test_loader.dataset)
            metrics_dict["net_global_eval_time"].append(time.time() - net_global_eval_start_time)
            metrics_dict["net_acc"].append(accuracy)

            metrics_dict["time_epoch_classifier"].append(time.time() - epoch_start_time)
            classifier_epochs_this_level += 1

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_no_improve = 0
                if level == 0:
                    best_classifier.load_state_dict(global_classifier.state_dict())
                else:
                    best_classifier.load_state_dict(classifier_head.state_dict(), strict=False)
            else:
                epochs_no_improve += 1
                print(f"Sem melhorias por {epochs_no_improve} épocas. Melhor acurácia: {best_accuracy:.4f}")

        global_classifier.load_state_dict(best_classifier.state_dict())

        # Salva métricas e finaliza o treino caso seja baseline
        if baseline:
            metrics_dict["time_level"].append(time.time() - level_start_time)
            try:
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
                print(f"Dicionário de métricas salvo em {metrics_path}")
            except Exception as e:
                print(f"Erro ao salvar o dicionário de métricas para JSON: {e}")
            
            checkpoint_data = {
                    'classifier_state_dict': global_classifier.state_dict(),
                }
            checkpoint_file = f"{experiment_dir}/checkpoint.pth"
            torch.save(checkpoint_data, checkpoint_file)
            print("Treinamento baseline concluído")
            break

        classifier_traffic_mb = CLASSIFIER_MODEL_SIZE_MB[dataset][level+1] * classifier_epochs_this_level
        metrics_dict["traffic_cost_classifier"].append(classifier_traffic_mb)

        metrics_dict["accuracy_transition"].append(accuracy)

        # --- TREINAMENTO DA GAN ---
        print(f"Alternando para o nível {level+1} de treinamento da GAN.")
        
        # Se for o último nível, ele deve pular a GAN e encerrar (como estava no seu if original "else: break")
        if level+1 > levels:
            metrics_dict["time_level"].append(time.time() - level_start_time)
            metrics_dict["MB_transmission"].append(classifier_traffic_mb)

            try:
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
                print(f"Dicionário de métricas salvo em {metrics_path}")
            except Exception as e:
                print(f"Erro ao salvar o dicionário de métricas para JSON: {e}")

            checkpoint_data = {'classifier_state_dict': global_classifier.state_dict(), 'level': level+1}
            torch.save(checkpoint_data, f"{experiment_dir}/checkpoint_end.pth")
            print("Máximo de níveis atingido. Treinamento finalizado!")
            break
            
        synthetic_dataset, classifier_head, feature_extractor = train_gan_for_level(
            args=args, 
            level=level, 
            models_dict=models_dict, 
            client_chunks=client_chunks, 
            global_classifier=global_classifier, 
            metrics_dict=metrics_dict, 
            device=device
        )
        
        # Atualiza o models_dict com as instâncias criadas na GAN para o próximo nível usar
        models_dict["classifier_head"] = classifier_head
        models_dict["feature_extractor"] = feature_extractor
        synthetic_dataset_state = {'generated_dataset': synthetic_dataset}

        # Computa métricas finais do nível (custo de rede, etc)
        discriminator_traffic_mb = DISCRIMINATOR_MODEL_SIZE_MB[dataset][level+1] * gan_epochs * num_chunks
        metrics_dict["traffic_cost_discriminator"].append(discriminator_traffic_mb)

        discriminator_embedding_transfer_mb = DISCRIMINATOR_EMBEDDING_TRANSFER_MB[dataset][level][0]
        classifier_synthetic_embedding_transfer_mb = CLASSIFIER_SYNTHETIC_EMBEDDING_TRANSFER_MB[dataset][args.num_syn][level][0]
        metrics_dict["x_fake_samples_size"].append(discriminator_embedding_transfer_mb)
        metrics_dict["syn_emb_ds"].append(classifier_synthetic_embedding_transfer_mb)
        metrics_dict["MB_transmission"].append(
            classifier_traffic_mb
            + discriminator_traffic_mb
            + discriminator_embedding_transfer_mb
            + classifier_synthetic_embedding_transfer_mb
        )
        metrics_dict["time_level"].append(time.time() - level_start_time)

        # Salva as métricas no final do loop
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
            
        # Salva o checkpoint com o dataset recém gerado
        checkpoint_data = {
                'classifier_state_dict': global_classifier.state_dict(),
                'level': level+1,
                'generated_dataset': synthetic_dataset,
            }
        checkpoint_file = f"{experiment_dir}/checkpoint_level{level+1}.pth"
        torch.save(checkpoint_data, checkpoint_file)

        print(f"Nível {level+1} concluído")

# --- BLOCO DE ORQUESTRAÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # 1. Obter argumentos
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"""Configuração do FLEG:
    Nível de checkpoint: {args.checkpoint_level}
    Conjunto de dados: {args.dataset}
    Número de chunks: {args.num_chunks}
    Número de partições: {args.num_partitions}
    Particionador: {args.partitioner}
    Estratégia: {args.strategy}
    Device: {device}
    """)

    # 2. Configurar o ambiente e inicializar redes/checkpoints
    partitioner, models_dict, start_level, experiment_dir, resume_checkpoint = setup_environment_and_models(args, device)
    
    # 3. Preparar os dados (Treino, Teste e Divisões)
    data_dict = prepare_datasets(args, partitioner, models_dict["image_key"])
    
    # 4. Executar o Loop Principal
    run_training_loop(args, models_dict, data_dict, start_level, experiment_dir, resume_checkpoint, device)
