import pandas as pd
class MedianCutSampling:
    def colors_and_counts(self,input_img):
        Implement This
        return pd.Series([tuple(x) for x in input_img.reshape(-1,3)]).value_counts()
    def quantize_image(self,input_img,k_popularity_algo= 512):
        Implement This
        top_colors = self.colors_and_counts(input_img)
        topk_colors= top_colors.iloc[0:k_popularity_algo]
        return topk_colors 