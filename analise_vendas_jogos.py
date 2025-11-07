import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

def carregar_e_limpar_dados(filepath='database/vgsales.csv'):
    print("=" * 80)
    print("CARREGANDO E LIMPANDO DADOS")
    print("=" * 80)
    df = pd.read_csv(filepath)
    print(f"\nShape inicial: {df.shape}")
    print(f"\nPrimeiras linhas:")
    print(df.head())

    print(f"\nInformações do dataset:")
    print(df.info())

    print(f"\nEstatísticas descritivas:")
    print(df.describe())

    print(f"\nValores nulos por coluna:")
    print(df.isnull().sum())

    df_clean = df.dropna(subset=['Year'])
    df_clean['Year'] = df_clean['Year'].astype(int)
    df_clean['Publisher'] = df_clean['Publisher'].fillna('Unknown')
    df_clean = df_clean[
        (df_clean['NA_Sales'] > 0) |
        (df_clean['EU_Sales'] > 0) |
        (df_clean['JP_Sales'] > 0) |
        (df_clean['Other_Sales'] > 0)
    ]
    if 'Global_Sales' not in df_clean.columns:
        df_clean['Global_Sales'] = (
            df_clean['NA_Sales'] +
            df_clean['EU_Sales'] +
            df_clean['JP_Sales'] +
            df_clean['Other_Sales']
        )
    df_clean['Decade'] = (df_clean['Year'] // 10) * 10
    df_clean['Success_Category'] = pd.cut(
        df_clean['Global_Sales'],
        bins=[0, 0.1, 0.5, 1, 5, 100],
        labels=['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Blockbuster']
    )

    print(f"\nShape após limpeza: {df_clean.shape}")
    print(f"\nLinhas removidas: {df.shape[0] - df_clean.shape[0]}")

    return df_clean

def analise_exploratoria(df):
    print("\n" + "=" * 80)
    print("ANÁLISE EXPLORATÓRIA")
    print("=" * 80)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise Exploratória de Vendas de Jogos de Vídeo',
                 fontsize=16, fontweight='bold')
    vendas_plataforma = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
    vendas_plataforma.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Top 10 Plataformas por Vendas Globais', fontweight='bold')
    axes[0, 0].set_xlabel('Plataforma')
    axes[0, 0].set_ylabel('Vendas Globais (milhões)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    vendas_genero = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    vendas_genero.plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Vendas Globais por Gênero', fontweight='bold')
    axes[0, 1].set_xlabel('Vendas Globais (milhões)')
    axes[0, 1].set_ylabel('Gênero')
    axes[0, 1].grid(axis='x', alpha=0.3)
    vendas_regiao = pd.DataFrame({
        'América do Norte': [df['NA_Sales'].sum()],
        'Europa': [df['EU_Sales'].sum()],
        'Japão': [df['JP_Sales'].sum()],
        'Outros': [df['Other_Sales'].sum()]
    }).T
    vendas_regiao.columns = ['Vendas']
    vendas_regiao.plot(kind='pie', y='Vendas', ax=axes[1, 0], autopct='%1.1f%%',
                       colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    axes[1, 0].set_title('Distribuição de Vendas por Região', fontweight='bold')
    axes[1, 0].set_ylabel('')
    vendas_ano = df.groupby('Year')['Global_Sales'].sum()
    axes[1, 1].plot(vendas_ano.index, vendas_ano.values, marker='o', linewidth=2, color='green')
    axes[1, 1].fill_between(vendas_ano.index, vendas_ano.values, alpha=0.3, color='green')
    axes[1, 1].set_title('Evolução de Vendas Globais ao Longo do Tempo', fontweight='bold')
    axes[1, 1].set_xlabel('Ano')
    axes[1, 1].set_ylabel('Vendas Globais (milhões)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('analise_exploratoria_vendas.png', dpi=300, bbox_inches='tight')
    print("\nGráfico salvo: analise_exploratoria_vendas.png")
    plt.show()
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Análise Detalhada com Seaborn', fontsize=16, fontweight='bold')
    top_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(10).index
    df_top = df[df['Platform'].isin(top_platforms)]
    vendas_matriz = df_top.groupby('Platform')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
    sns.heatmap(vendas_matriz, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes2[0, 0])
    axes2[0, 0].set_title('Vendas por Região e Plataforma (Top 10)', fontweight='bold')
    axes2[0, 0].set_ylabel('Plataforma')
    sns.boxplot(data=df, x='Genre', y='Global_Sales', ax=axes2[0, 1])
    axes2[0, 1].set_title('Distribuição de Vendas por Gênero', fontweight='bold')
    axes2[0, 1].set_xlabel('Gênero')
    axes2[0, 1].set_ylabel('Vendas Globais (milhões)')
    axes2[0, 1].tick_params(axis='x', rotation=45)
    axes2[0, 1].set_ylim(0, 5)
    df_decade = df[df['Decade'] >= 1980].copy()
    df_decade['Decade'] = df_decade['Decade'].astype(str)
    sns.violinplot(data=df_decade, x='Decade', y='Global_Sales', ax=axes2[1, 0])
    axes2[1, 0].set_title('Distribuição de Vendas por Década', fontweight='bold')
    axes2[1, 0].set_ylabel('Vendas Globais (milhões)')
    axes2[1, 0].set_ylim(0, 5)
    top_platforms_count = df['Platform'].value_counts().head(10)
    sns.barplot(x=top_platforms_count.values, y=top_platforms_count.index,
                palette='viridis', ax=axes2[1, 1])
    axes2[1, 1].set_title('Top 10 Plataformas por Quantidade de Jogos', fontweight='bold')
    axes2[1, 1].set_xlabel('Quantidade de Jogos')
    axes2[1, 1].set_ylabel('Plataforma')
    plt.tight_layout()
    plt.savefig('analise_seaborn_vendas.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo: analise_seaborn_vendas.png")
    plt.show()

def criar_visualizacoes_interativas(df):
    print("\n" + "=" * 80)
    print("CRIANDO VISUALIZAÇÕES INTERATIVAS")
    print("=" * 80)
    vendas_tempo_regiao = df.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=vendas_tempo_regiao['Year'], y=vendas_tempo_regiao['NA_Sales'],
                              mode='lines+markers', name='América do Norte',
                              line=dict(width=2), marker=dict(size=6)))
    fig1.add_trace(go.Scatter(x=vendas_tempo_regiao['Year'], y=vendas_tempo_regiao['EU_Sales'],
                              mode='lines+markers', name='Europa',
                              line=dict(width=2), marker=dict(size=6)))
    fig1.add_trace(go.Scatter(x=vendas_tempo_regiao['Year'], y=vendas_tempo_regiao['JP_Sales'],
                              mode='lines+markers', name='Japão',
                              line=dict(width=2), marker=dict(size=6)))
    fig1.add_trace(go.Scatter(x=vendas_tempo_regiao['Year'], y=vendas_tempo_regiao['Other_Sales'],
                              mode='lines+markers', name='Outros',
                              line=dict(width=2), marker=dict(size=6)))

    fig1.update_layout(
        title='Evolução de Vendas ao Longo do Tempo por Região',
        xaxis_title='Ano',
        yaxis_title='Vendas (milhões)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )

    fig1.write_html('evolucao_vendas_regiao.html')
    print("\nGráfico interativo salvo: evolucao_vendas_regiao.html")
    fig1.show()
    top_games = df.nlargest(20, 'Global_Sales')
    fig2 = px.bar(top_games,
                  x='Global_Sales',
                  y='Name',
                  color='Platform',
                  title='Top 20 Jogos Mais Vendidos',
                  labels={'Global_Sales': 'Vendas Globais (milhões)', 'Name': 'Jogo'},
                  orientation='h',
                  hover_data=['Year', 'Genre', 'Publisher'],
                  height=700)
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig2.write_html('top_20_jogos.html')
    print("Gráfico interativo salvo: top_20_jogos.html")
    fig2.show()
    top_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(8).index
    df_sunburst = df[df['Platform'].isin(top_platforms)].copy()
    df_agg = df_sunburst.groupby(['Platform', 'Genre', 'Publisher'])['Global_Sales'].sum().reset_index()
    df_agg = df_agg.nlargest(100, 'Global_Sales')

    fig3 = px.sunburst(df_agg,
                       path=['Platform', 'Genre', 'Publisher'],
                       values='Global_Sales',
                       title='Hierarquia de Vendas: Plataforma → Gênero → Publisher',
                       height=800)

    fig3.write_html('sunburst_vendas.html')
    print("Gráfico interativo salvo: sunburst_vendas.html")
    fig3.show()
    top5_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(5).index
    df_scatter = df[df['Platform'].isin(top5_platforms)]
    fig4 = px.scatter(df_scatter,
                      x='Year',
                      y='Global_Sales',
                      color='Platform',
                      size='Global_Sales',
                      hover_data=['Name', 'Genre', 'Publisher'],
                      title='Vendas ao Longo do Tempo por Plataforma (Top 5)',
                      labels={'Global_Sales': 'Vendas Globais (milhões)', 'Year': 'Ano'},
                      height=600)
    fig4.update_layout(template='plotly_white')
    fig4.write_html('scatter_vendas_ano.html')
    print("Gráfico interativo salvo: scatter_vendas_ano.html")
    fig4.show()
    top_publishers = df.groupby('Publisher')['Global_Sales'].sum().nlargest(20).reset_index()
    fig5 = px.treemap(top_publishers,
                      path=['Publisher'],
                      values='Global_Sales',
                      title='Market Share - Top 20 Publishers',
                      height=600)
    fig5.update_traces(textinfo='label+value+percent parent')
    fig5.write_html('treemap_publishers.html')
    print("Gráfico interativo salvo: treemap_publishers.html")
    fig5.show()

def criar_histogramas(df):
    print("\n" + "=" * 80)
    print("CRIANDO HISTOGRAMAS")
    print("=" * 80)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Histogramas de Distribuição de Vendas', fontsize=16, fontweight='bold')

    axes[0, 0].hist(df['Global_Sales'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribuição de Vendas Globais', fontweight='bold')
    axes[0, 0].set_xlabel('Vendas Globais (milhões)')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].axvline(df['Global_Sales'].mean(), color='red', linestyle='--',
                       label=f'Média: {df["Global_Sales"].mean():.2f}')
    axes[0, 0].axvline(df['Global_Sales'].median(), color='green', linestyle='--',
                       label=f'Mediana: {df["Global_Sales"].median():.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    vendas_por_ano = df.groupby('Year')['Global_Sales'].sum()
    axes[0, 1].bar(vendas_por_ano.index, vendas_por_ano.values, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Vendas Totais por Ano de Lançamento', fontweight='bold')
    axes[0, 1].set_xlabel('Ano')
    axes[0, 1].set_ylabel('Vendas Totais (milhões)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    top_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(10)
    axes[1, 0].barh(range(len(top_platforms)), top_platforms.values, color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(len(top_platforms)))
    axes[1, 0].set_yticklabels(top_platforms.index)
    axes[1, 0].set_title('Top 10 Plataformas - Vendas Totais', fontweight='bold')
    axes[1, 0].set_xlabel('Vendas Totais (milhões)')
    axes[1, 0].grid(axis='x', alpha=0.3)
    jogos_por_ano = df.groupby('Year').size()
    axes[1, 1].bar(jogos_por_ano.index, jogos_por_ano.values, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Quantidade de Jogos Lançados por Ano', fontweight='bold')
    axes[1, 1].set_xlabel('Ano')
    axes[1, 1].set_ylabel('Quantidade de Jogos')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('histogramas_vendas.png', dpi=300, bbox_inches='tight')
    print("\nGráfico salvo: histogramas_vendas.png")
    plt.show()
    fig_interactive = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribuição de Vendas por Plataforma (Top 10)',
                        'Distribuição de Vendas por Gênero',
                        'Jogos Lançados por Década',
                        'Distribuição de Vendas por Categoria de Sucesso')
    )
    top10_plat = df.groupby('Platform')['Global_Sales'].sum().nlargest(10).reset_index()
    fig_interactive.add_trace(
        go.Bar(x=top10_plat['Platform'], y=top10_plat['Global_Sales'], name='Plataforma'),
        row=1, col=1
    )
    vendas_genre = df.groupby('Genre')['Global_Sales'].sum().reset_index()
    fig_interactive.add_trace(
        go.Bar(x=vendas_genre['Genre'], y=vendas_genre['Global_Sales'], name='Gênero'),
        row=1, col=2
    )
    jogos_decade = df.groupby('Decade').size().reset_index(name='Count')
    fig_interactive.add_trace(
        go.Bar(x=jogos_decade['Decade'], y=jogos_decade['Count'], name='Década'),
        row=2, col=1
    )
    success_counts = df['Success_Category'].value_counts().reset_index()
    success_counts.columns = ['Category', 'Count']
    fig_interactive.add_trace(
        go.Bar(x=success_counts['Category'], y=success_counts['Count'], name='Sucesso'),
        row=2, col=2
    )

    fig_interactive.update_layout(
        title_text='Histogramas Interativos - Análise de Vendas de Jogos',
        showlegend=False,
        height=800
    )

    fig_interactive.write_html('histogramas_interativos.html')
    print("Gráfico interativo salvo: histogramas_interativos.html")
    fig_interactive.show()

def gerar_insights(df):
    print("\n" + "=" * 80)
    print("INSIGHTS E ESTATÍSTICAS")
    print("=" * 80)

    print("\n1. TOP 5 JOGOS MAIS VENDIDOS:")
    top5_games = df.nlargest(5, 'Global_Sales')[['Name', 'Platform', 'Year', 'Genre', 'Global_Sales']]
    print(top5_games.to_string(index=False))

    print("\n2. TOP 5 PLATAFORMAS POR VENDAS:")
    top5_platforms = df.groupby('Platform')['Global_Sales'].sum().nlargest(5)
    for platform, sales in top5_platforms.items():
        print(f"   {platform}: {sales:.2f} milhões")

    print("\n3. TOP 5 GÊNEROS POR VENDAS:")
    top5_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(5)
    for genre, sales in top5_genres.items():
        print(f"   {genre}: {sales:.2f} milhões")

    print("\n4. TOP 5 PUBLISHERS POR VENDAS:")
    top5_publishers = df.groupby('Publisher')['Global_Sales'].sum().nlargest(5)
    for publisher, sales in top5_publishers.items():
        print(f"   {publisher}: {sales:.2f} milhões")

    print("\n5. ANÁLISE POR REGIÃO:")
    print(f"   América do Norte: {df['NA_Sales'].sum():.2f} milhões ({df['NA_Sales'].sum()/df['Global_Sales'].sum()*100:.1f}%)")
    print(f"   Europa: {df['EU_Sales'].sum():.2f} milhões ({df['EU_Sales'].sum()/df['Global_Sales'].sum()*100:.1f}%)")
    print(f"   Japão: {df['JP_Sales'].sum():.2f} milhões ({df['JP_Sales'].sum()/df['Global_Sales'].sum()*100:.1f}%)")
    print(f"   Outros: {df['Other_Sales'].sum():.2f} milhões ({df['Other_Sales'].sum()/df['Global_Sales'].sum()*100:.1f}%)")

    print("\n6. ANÁLISE TEMPORAL:")
    best_year = df.groupby('Year')['Global_Sales'].sum().idxmax()
    best_year_sales = df.groupby('Year')['Global_Sales'].sum().max()
    print(f"   Melhor ano: {best_year} com {best_year_sales:.2f} milhões em vendas")

    most_releases = df.groupby('Year').size().idxmax()
    num_releases = df.groupby('Year').size().max()
    print(f"   Ano com mais lançamentos: {most_releases} com {num_releases} jogos")

    print("\n7. ESTATÍSTICAS GERAIS:")
    print(f"   Total de jogos: {len(df)}")
    print(f"   Total de plataformas: {df['Platform'].nunique()}")
    print(f"   Total de gêneros: {df['Genre'].nunique()}")
    print(f"   Total de publishers: {df['Publisher'].nunique()}")
    print(f"   Vendas globais totais: {df['Global_Sales'].sum():.2f} milhões")
    print(f"   Média de vendas por jogo: {df['Global_Sales'].mean():.2f} milhões")
    print(f"   Mediana de vendas por jogo: {df['Global_Sales'].median():.2f} milhões")

def main():
    print("\n" + "=" * 80)
    print("ANÁLISE DE VENDAS DE JOGOS DE VÍDEO")
    print("Desafio 8 - Análise de Dados Kaggle")
    print("=" * 80)
    filepath = 'database/vgsales.csv'
    print(f"\nArquivo de dados: {filepath}")
    print("\nIniciando análise...")

    try:
        df = carregar_e_limpar_dados(filepath)
        analise_exploratoria(df)
        criar_visualizacoes_interativas(df)
        criar_histogramas(df)
        gerar_insights(df)
        df.to_csv('database/vgsales_clean.csv', index=False)
        print("\n" + "=" * 80)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print("\nArquivos gerados:")
        print("  - vgsales_clean.csv (dados limpos)")
        print("  - analise_exploratoria_vendas.png")
        print("  - analise_seaborn_vendas.png")
        print("  - histogramas_vendas.png")
        print("  - evolucao_vendas_regiao.html")
        print("  - top_20_jogos.html")
        print("  - sunburst_vendas.html")
        print("  - scatter_vendas_ano.html")
        print("  - treemap_publishers.html")
        print("  - histogramas_interativos.html")

    except FileNotFoundError:
        print(f"\nERRO: Arquivo '{filepath}' não encontrado!")
        print("\nPor favor, baixe o dataset de:")
        print("https://www.kaggle.com/datasets/gregorut/videogamesales")
        print("\nE coloque o arquivo 'vgsales.csv' no diretório 'database'.")
    except Exception as e:
        print(f"\nERRO: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
