# Explainable Algorithmic Trading Agent via Symbolic Regression

This paper presents `EATA`, by using symbolic regression to conformally predict the future price distribution, thus creating the trading signals.

## Architecture

下面基于项目实际代码关系给出两张 Mermaid 架构图：模块依赖图与训练/预测时序图。你可以将代码块粘贴到 Mermaid 预览器或支持 Mermaid 的 Markdown 工具中渲染。

## 模块依赖关系图

```mermaid
graph TD
    subgraph Project Root
        README[README.md]
        QQE[QQE.py]
        SW[sliding_window_nemots.py]
        Adapter[nemots_adapter.py]
        TestDir[Test/ ...]
    end

    subgraph nemots
        NEngine[engine.py Engine, OptimizedMetrics]
        NModel[model.py Model]
        NMCTS[mcts.py MCTS]
        NMCTSAdp[mcts_adapter.py MCTSAdapter]
        NNet[network.py PVNetCtx]
        NSym[symbolics.py grammar & functions]
        NScore[score.py score_with_est, simplify_eq]
        NArgs[args.py Args]
        NTrack[tracker.py Tracker]
        NMetrics[OptimizedMetrics]
    end

    Adapter --> |uses| NPredict[NEMoTSPredictor in nemots_adapter.py]
    NPredict --> |optional| Full[FullNEMoTSAdapter]
    NPredict --> |fallback| Simple[SimpleNEMoTS]

    Full --> |.fit/.predict_action| NEngine
    Full --> |prepare data| Adapter
    NEngine --> |run| NModel
    NModel --> |search| NMCTS
    NModel --> |policy/value| NNet
    NModel --> |grammar| NSym
    NModel --> |score| NScore
    NModel --> |records| NTrack
    NMCTSAdp --> |patch| NMCTS
    NEngine --> NMetrics

    SW --> |Engine + Args| NEngine
    SW --> |hyperparams| NArgs

    Simple --> |random expr| Adapter
    Simple --> |score| Adapter
```

关键关系引用：

- [NEMoTSAdapter](cci:2://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:666:0-706:9) 统一入口，内部的 [NEMoTSPredictor](cci:2://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:461:0-664:39) 根据数据量选择：
  - 充足数据：[FullNEMoTSAdapter](cci:2://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:226:0-459:38) 调用 [nemots.engine.Engine.simulate()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/engine.py:26:4-52:80) → [model.Model.run()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/mcts.py:237:4-357:79) → [mcts.MCTS](cci:2://file:///Users/yin/Desktop/doing/eata/nemots/mcts.py:7:0-365:36) + `network.PVNetCtx`，并用 [mcts_adapter.MCTSAdapter](cci:2://file:///Users/yin/Desktop/doing/eata/nemots/mcts_adapter.py:17:0-165:24) 对齐维度，打分经 [score.score_with_est](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/score.py:48:0-136:16) 与 [engine.OptimizedMetrics.metrics](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/engine.py:164:4-280:39)。
  - 数据不足：[SimpleNEMoTS](cci:2://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:132:0-224:40) 在 [nemots_adapter.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:0:0-0:0) 内部生成与评估符号表达式。
- [sliding_window_nemots.py](cci:7://file:///Users/yin/Desktop/doing/eata/sliding_window_nemots.py:0:0-0:0) 的 [SlidingWindowNEMoTS](cci:2://file:///Users/yin/Desktop/doing/eata/sliding_window_nemots.py:16:0-291:9) 直接构造 [Engine(Args)](cci:2://file:///Users/yin/Desktop/doing/eata/nemots/engine.py:16:0-160:12)，使用滑窗数据并将前一窗最佳表达式以继承方式传入 [simulate()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/engine.py:26:4-52:80)。

## 训练/预测时序图（完整 NEMoTS 路径）

```mermaid
sequenceDiagram
    autonumber
    participant User as User Code
    participant Adapter as NEMoTSAdapter
    participant Pred as NEMoTSPredictor
    participant Full as FullNEMoTSAdapter
    participant Eng as nemots.Engine
    participant Mod as nemots.Model
    participant MCTS as nemots.MCTS (+ MCTSAdapter)
    participant Net as PVNetCtx (network)
    participant Score as score/metrics

    User->>Adapter: train(df)
    Adapter->>Pred: .fit(df)
    Pred->>Full: 构造 + .fit(df)
    Full->>Full: _prepare_training_data(df) → (X,y)
    Full->>Full: _convert_to_nemots_format(X,y) → data
    Full->>Eng: simulate(data, inherited_tree?)

    Eng->>Mod: run(X, y, inherited_tree)
    Mod->>MCTS: 构造(MCTSAdapter.patch_mcts)
    loop num_transplant × num_runs
        MCTS->>Net: get_policy3(...) → policy_nn, value
        MCTS->>MCTS: 融合 NN policy 与 UCB
        MCTS->>MCTS: 搜索/rollout/回传
        MCTS->>Mod: 返回 best_solution, records
        Mod->>Mod: 更新 data_buffer / aug_grammars
    end
    Mod-->>Eng: all_eqs, test_scores, supervision_data, policy, reward
    Eng->>Score: OptimizedMetrics.metrics(...)
    Score-->>Eng: mae, mse, corr, best_exp
    Eng-->>Full: 返回(best_exp, mae, mse, corr, ...)

    Full-->>Pred: is_trained=True
    Pred-->>Adapter: done

    User->>Adapter: predict(df)
    Adapter->>Pred: .predict_action(df)
    alt FullNEMoTSAdapter
        Pred->>Full: .predict_action(df)
        Full-->>Pred: action {-1,0,1}
    else SimpleNEMoTS
        Pred->>Pred: 简化表达式预测
        Pred-->>Adapter: action {-1,0,1}
    end
```

## 补充说明

- **顶层入口**：
  
  - [nemots_adapter.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:0:0-0:0) 提供统一接口：[NEMoTSAdapter.train()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:676:4-686:24), [NEMoTSAdapter.predict()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:688:4-693:48)。
  - [sliding_window_nemots.py](cci:7://file:///Users/yin/Desktop/doing/eata/sliding_window_nemots.py:0:0-0:0) 提供滑窗增强版本：[SlidingWindowNEMoTS.sliding_fit()](cci:1://file:///Users/yin/Desktop/doing/eata/sliding_window_nemots.py:160:4-220:13), [SlidingWindowNEMoTS.predict()](cci:1://file:///Users/yin/Desktop/doing/eata/sliding_window_nemots.py:222:4-272:20)，直接用 [Engine](cci:2://file:///Users/yin/Desktop/doing/eata/nemots/engine.py:16:0-160:12) 与 [Args](cci:2://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:237:8-297:90)。

- **核心搜索与评估链路**：
  
  - [nemots/model.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots/model.py:0:0-0:0) 中 [Model.run()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/model.py:69:4-252:80) 负责构建 [MCTS](cci:2://file:///Users/yin/Desktop/doing/eata/nemots/mcts.py:7:0-365:36)，组织 grammar（[symbolics.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots/symbolics.py:0:0-0:0)），引导搜索（`network.PVNetCtx`），并将搜索轨迹缓存到 `data_buffer`。
  - [nemots/mcts_adapter.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots/mcts_adapter.py:0:0-0:0) 的 [MCTSAdapter.patch_mcts()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/mcts_adapter.py:70:4-126:28) 对 [MCTS.get_policy3()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/mcts.py:207:4-214:28) 进行维度对齐，保证 NN 策略输出与 MCTS 语法空间一致。
  - [nemots/score.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots/score.py:0:0-0:0) 的 [score_with_est()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/score.py:48:0-136:16) 对表达式进行系数估计与评分，[nemots/engine.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots/engine.py:0:0-0:0) 的 [OptimizedMetrics.metrics()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots/engine.py:164:4-280:39) 用于最终度量与表达式选择。

- **简化路径**：
  
  - [SimpleNEMoTS](cci:2://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:132:0-224:40) 在 [nemots_adapter.py](cci:7://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:0:0-0:0) 中，通过随机表达式模板与 [StockScorer.score_expression()](cci:1://file:///Users/yin/Desktop/doing/eata/nemots_adapter.py:71:4-130:34) 简化评估，数据不足时兜底。

<style>#mermaid-1759116750117{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1759116750117 .error-icon{fill:#552222;}#mermaid-1759116750117 .error-text{fill:#552222;stroke:#552222;}#mermaid-1759116750117 .edge-thickness-normal{stroke-width:2px;}#mermaid-1759116750117 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1759116750117 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1759116750117 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1759116750117 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1759116750117 .marker{fill:#333333;}#mermaid-1759116750117 .marker.cross{stroke:#333333;}#mermaid-1759116750117 svg{font-family:sans-serif;font-size:16px;}#mermaid-1759116750117 .label{font-family:sans-serif;color:#333;}#mermaid-1759116750117 .label text{fill:#333;}#mermaid-1759116750117 .node rect,#mermaid-1759116750117 .node circle,#mermaid-1759116750117 .node ellipse,#mermaid-1759116750117 .node polygon,#mermaid-1759116750117 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1759116750117 .node .label{text-align:center;}#mermaid-1759116750117 .node.clickable{cursor:pointer;}#mermaid-1759116750117 .arrowheadPath{fill:#333333;}#mermaid-1759116750117 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1759116750117 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1759116750117 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1759116750117 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1759116750117 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1759116750117 .cluster text{fill:#333;}#mermaid-1759116750117 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1759116750117:root{--mermaid-font-family:sans-serif;}#mermaid-1759116750117:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1759116750117 flowchart{fill:apa;}</style>
