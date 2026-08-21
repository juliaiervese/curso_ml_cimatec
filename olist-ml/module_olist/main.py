from pathlib import Path

from loguru import logger

from module_olist.dataset import (
    load_data,
    create_dataset,
    save_dataset,
)

from module_olist.features import (
    create_features,
)


def main():

    logger.info("Starting Olist ML pipeline")


    project_root = Path(__file__).resolve().parents[1]

    raw_path = project_root / "data" / "raw"
 
    interim_path = (
        project_root
        / "data"
        / "interim"
        / "olist_interim.csv"
    )


    # --------------------------------------------
    # 1. Load raw data
    # --------------------------------------------

    orders, items, customers = load_data(
        orders_path=raw_path / "olist_orders_dataset.csv",
        items_paths=raw_path / "olist_order_items_dataset.csv",
        customers_path=raw_path / "olist_customers_dataset.csv",
    )


    # --------------------------------------------
    # 2. Create dataset
    # --------------------------------------------

    dataset = create_dataset(
        orders=orders,
        items=items,
        customers=customers,
    )


    logger.info(
        f"Dataset after merge: {dataset.shape}"
    )


    # --------------------------------------------
    # 3. Create features
    # --------------------------------------------

    dataset_features = create_features(
        dataset
    )


    logger.info(
        f"Dataset after features: {dataset_features.shape}"
    )


    # --------------------------------------------
    # 4. Save interim with features
    # --------------------------------------------

    interim_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    save_dataset(
        dataset=dataset_features,
        output_path=interim_path,
    )


    logger.success(
        "Olist pipeline completed successfully"
    )


if __name__ == "__main__":
    main()